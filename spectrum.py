"""Spectrum state management, fast-forward path, and shared sampling logic."""

import logging
import math
from typing import Optional, Dict

import torch

import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview

from .forecaster import SpectrumPredictor

_LOG_SIGMA_EPS = 1e-8

# Samplers we forbid two consecutive cached steps for. Two distinct failure
# modes both benefit from the guard:
#   * Multistep / SDE samplers compute finite differences across `denoised`
#     values from previous steps (dpmpp_2m: denoised_d; dpmpp_3m_sde + er_sde:
#     denoised_d + denoised_u). Tight sigma gaps divide prediction errors by
#     near-zero and diverge.
#   * Single-step ancestral samplers compound cache errors through
#     closed-loop feedback: drifted x → wrong next denoised(x) → ancestral
#     noise injected with off-trajectory scale → random-walk amplification.
#     Validated by `bench/spectrum/analyze_drift.py` in the anima_lora repo —
#     adding euler_ancestral / dpmpp_2s_ancestral to the guard cuts drift on
#     karras / exponential / kl_optimal by 10–30%.
_FRAGILE_SAMPLERS = frozenset({
    # Multistep / SDE FD-amplification group
    "dpmpp_sde", "dpmpp_sde_gpu",
    "dpmpp_2m", "dpmpp_2m_cfg_pp",
    "dpmpp_2m_sde", "dpmpp_2m_sde_gpu",
    "dpmpp_2m_sde_heun", "dpmpp_2m_sde_heun_gpu",
    "dpmpp_3m_sde", "dpmpp_3m_sde_gpu",
    "ipndm", "ipndm_v", "deis", "lms",
    "res_multistep", "res_multistep_cfg_pp",
    "res_multistep_ancestral", "res_multistep_ancestral_cfg_pp",
    "er_sde", "sa_solver", "sa_solver_pece",
    # Ancestral feedback group
    "euler_ancestral", "euler_ancestral_cfg_pp",
    "dpm_2_ancestral",
    "dpmpp_2s_ancestral", "dpmpp_2s_ancestral_cfg_pp",
})
# Backwards-compat alias — older code may import the previous name.
_FRAGILE_MULTISTEP_SAMPLERS = _FRAGILE_SAMPLERS

logger = logging.getLogger(__name__)


def _spectrum_fast_forward(
    dit, timestep: torch.Tensor, predicted_feature: torch.Tensor
) -> torch.Tensor:
    """Runs only t_embedder + final_layer + unpatchify on predicted features.

    Returns the same shape as diffusion_model.forward() — 5D for video DiTs.
    """
    if timestep.ndim == 1:
        timestep = timestep.unsqueeze(1)
    # Replicate the model's two-step t_embedder call: Timesteps (sinusoidal,
    # always float32) -> cast to model dtype -> TimestepEmbedding (linear layers).
    # Calling t_embedder as a single Sequential skips the intermediate cast.
    t_sinusoidal = dit.t_embedder[0](timestep)
    t_emb, adaln = dit.t_embedder[1](t_sinusoidal.to(predicted_feature.dtype))
    t_emb = dit.t_embedding_norm(t_emb)
    # Mod guidance: add cached pooled-text projection from the DIFFUSION_MODEL
    # wrapper.  On actual steps the wrapper computes base+delta from post-adapter
    # context and caches it on dit._mod_pooled_proj.  On cached steps we reuse
    # the last actual step's value (text doesn't change between steps).
    pooled_proj = getattr(dit, "_mod_pooled_proj", None)
    if pooled_proj is not None:
        pp = pooled_proj.unsqueeze(1).to(t_emb.dtype)
        if pp.shape[0] == t_emb.shape[0]:
            t_emb = t_emb + pp
        elif pp.shape[0] == 1:
            t_emb = t_emb + pp.expand_as(t_emb)
    x = dit.final_layer(predicted_feature, t_emb, adaln_lora_B_T_3D=adaln)
    return dit.unpatchify(x)


class SpectrumState:
    def __init__(
        self,
        window_size: float,
        flex_window: float,
        warmup_steps: int,
        w: float,
        m: int,
        lam: float,
        num_steps: int,
        stop_caching_step: int,
        log_sigma_min: float,
        log_sigma_max: float,
        delta_ls_unit: float,
        forbid_consecutive_cache: bool,
    ):
        self.window_size = window_size
        self.flex_window = flex_window
        self.warmup_steps = warmup_steps
        self.w = w
        self.m_param = m
        self.lam = lam
        self.num_steps = num_steps
        self.stop_caching_step = stop_caching_step
        self.log_sigma_min = log_sigma_min
        self.log_sigma_max = log_sigma_max
        self.delta_ls_unit = delta_ls_unit
        self.forbid_consecutive_cache = forbid_consecutive_cache

        # Log-sigma trigger threshold. Grows by flex_window units after each
        # actual forward — same "trust predictor more later" intent as the old
        # curr_ws ramp, just measured in log-sigma instead of step count.
        self.delta_ls = window_size * delta_ls_unit
        self.delta_ls_growth = flex_window * delta_ls_unit
        self.last_actual_log_sigma: Optional[float] = None

        # Runtime
        self.step_idx = -1
        self.last_sigma: Optional[float] = None
        self.last_sigma_log: Optional[float] = None
        self.mode = "actual"
        self.consec_cached = 0
        self.fwd_count = 0

        # Forecasters keyed by cond_or_uncond value (0=cond, 1=uncond)
        self.forecasters: Dict[int, SpectrumPredictor] = {}
        self.captured_feat: Optional[torch.Tensor] = None

    def should_cache(self, current_log_sigma: float) -> bool:
        if self.step_idx < self.warmup_steps:
            return False
        if self.step_idx >= self.stop_caching_step:
            return False
        # Guard for fragile multistep samplers: never two cached in a row so
        # their (denoised[i] - denoised[i-1]) finite differences have at most
        # one predicted operand — prevents error-over-tiny-Δλ explosion.
        if self.forbid_consecutive_cache and self.consec_cached >= 1:
            return False
        if self.last_actual_log_sigma is None:
            return False
        # Cache while the log-sigma distance from the last actual is below the
        # current threshold. Produces a log-sigma-uniform actual/cached pattern
        # regardless of scheduler shape.
        return abs(current_log_sigma - self.last_actual_log_sigma) < self.delta_ls

    def note_actual(self, log_sigma: float) -> None:
        self.last_actual_log_sigma = log_sigma
        # Match the old curr_ws ramp: only grow the threshold once we're past
        # warmup, so initial warmup actuals don't inflate it prematurely.
        if self.step_idx >= self.warmup_steps:
            self.delta_ls += self.delta_ls_growth

    def has_forecasters(self, cond_or_uncond: list) -> bool:
        return all(cou in self.forecasters for cou in cond_or_uncond)


def _capture_pre_hook(module, args):
    """Module-singleton pre-hook on final_layer — stores the pre-final feature
    on whichever SpectrumState is currently bound to the module.
    """
    state = getattr(module, "_spectrum_state", None)
    if state is not None:
        state.captured_feat = args[0].detach().clone()


def _resolve_schedule_stats(
    model_sampling, scheduler: str, steps: int, warmup_steps: int, stop_caching_step: int
):
    """Return (log_sigma_min, log_sigma_max, delta_ls_unit).

    `delta_ls_unit` is the mean |Δ log_sigma| per step across the caching
    region (warmup..stop-1). Used to convert window_size/flex_window (step-
    count knobs) into log-sigma thresholds, so the actual/cached schedule
    spreads forwards uniformly in log-sigma rather than step index. This is
    what makes karras / exponential / kl_optimal behave — on those, step-
    uniform actuals all cluster in the high-sigma shoulder, leaving the
    detail-forming tail unsampled.
    """
    log_sigmas = None
    try:
        sigmas = comfy.samplers.calculate_sigmas(model_sampling, scheduler, steps)
        nonzero = sigmas[sigmas > _LOG_SIGMA_EPS]
        if nonzero.numel() >= 2:
            log_sigmas = torch.log(nonzero).float().cpu()
            log_sigma_min = float(log_sigmas.min().item())
            log_sigma_max = float(log_sigmas.max().item())
        else:
            raise ValueError("not enough non-zero sigmas")
    except Exception as e:
        logger.warning(
            f"Spectrum: calculate_sigmas failed for scheduler={scheduler!r} ({e}); "
            "falling back to model_sampling sigma bounds (step-uniform)."
        )
        sig_min = max(float(model_sampling.sigma_min), _LOG_SIGMA_EPS)
        sig_max = max(float(model_sampling.sigma_max), sig_min * 10.0)
        log_sigma_min = math.log(sig_min)
        log_sigma_max = math.log(sig_max)
        # No schedule — use uniform log-sigma unit.
        delta_ls_unit = (log_sigma_max - log_sigma_min) / max(1, steps - 1)
        return log_sigma_min, log_sigma_max, delta_ls_unit

    # delta_ls_unit: mean absolute log-sigma gap across the caching region.
    # Clip to [warmup, stop] so we measure the region where the heuristic will
    # actually operate. On front-loaded schedulers this is the fine-grained tail.
    lo = max(0, min(warmup_steps, log_sigmas.numel() - 1))
    hi = max(lo + 2, min(stop_caching_step + 1, log_sigmas.numel()))
    region = log_sigmas[lo:hi]
    if region.numel() >= 2:
        delta_ls_unit = float((region[1:] - region[:-1]).abs().mean().item())
    else:
        delta_ls_unit = (log_sigma_max - log_sigma_min) / max(1, steps - 1)
    delta_ls_unit = max(delta_ls_unit, 1e-6)
    return log_sigma_min, log_sigma_max, delta_ls_unit


def _auto_stop_caching_step(
    model_sampling, scheduler: str, steps: int, base_keep: int = 3
) -> int:
    """Schedule-aware stop_caching_step from the late log-σ tail-snap ratio.

    Schedules with a strong late-Δλ tail (simple, sgm_uniform) self-correct
    accumulated cache error in the last few steps because the huge final
    log-σ jump pulls x straight to denoised. Uniform schedules (karras /
    exponential / kl_optimal) lack this wash-out, so any mid-schedule cache
    error persists to the endpoint as visible drift.

    Heuristic: ratio = mean(last 3 log-σ gaps) / mean(all log-σ gaps).
        ratio ≥ 2 → strong tail-snap, keep base_keep actuals at the end.
        ratio < 2 → uniform, push stop earlier so more late actuals dilute
                    accumulated drift (extra ≈ round(5 · (2 − ratio))).
    Validated against `bench/spectrum/analyze_drift.py` (anima_lora repo) —
    cuts final drift on karras/exp/kl by 15–25% with no quality cost on
    simple. See README "stop_caching_step = -1" auto-mode notes.
    """
    try:
        sigmas = comfy.samplers.calculate_sigmas(model_sampling, scheduler, steps)
        nonzero = sigmas[sigmas > _LOG_SIGMA_EPS]
        if nonzero.numel() < 5:
            return max(0, steps - base_keep)
        log_s = torch.log(nonzero).float().cpu()
        gaps = (log_s[1:] - log_s[:-1]).abs()
        if gaps.numel() < 4:
            return max(0, steps - base_keep)
        tail3 = float(gaps[-3:].mean().item())
        overall = float(gaps.mean().item())
        ratio = tail3 / max(overall, 1e-6)
        if ratio >= 2.0:
            keep = base_keep
        else:
            keep = base_keep + max(1, int(round(5.0 * (2.0 - ratio))))
        return max(0, steps - keep)
    except Exception as e:
        logger.warning(
            f"Spectrum: auto stop_caching_step failed ({e}); falling back to "
            f"steps - {base_keep}."
        )
        return max(0, steps - base_keep)


def _ensure_capture_hook(dit) -> None:
    final_layer = dit.final_layer
    if getattr(final_layer, "_spectrum_hook_installed", False):
        return
    final_layer.register_forward_pre_hook(_capture_pre_hook)
    final_layer._spectrum_hook_installed = True


def spectrum_sample(
    model,
    seed,
    steps,
    cfg,
    sampler_name,
    scheduler,
    positive,
    negative,
    latent_image,
    denoise,
    window_size,
    flex_window,
    warmup_steps,
    blend_w,
    cheby_degree,
    ridge_lambda,
    stop_caching_step=-1,
):
    """Shared Spectrum sampling logic used by all node tiers."""
    m = model.clone()

    dit = m.model.diffusion_model
    model_sampling = m.model.model_sampling

    # stop_caching_step: -1 = auto. Picks based on the schedule's late-Δλ
    # tail-snap ratio — uniform schedules (karras / exponential / kl_optimal)
    # get more late actuals to dilute drift, simple keeps the prior `steps-3`
    # behavior for full speedup. Clamped to [warmup_steps, steps].
    if stop_caching_step < 0:
        stop_caching_step = _auto_stop_caching_step(model_sampling, scheduler, steps)
    stop_caching_step = max(warmup_steps, min(stop_caching_step, steps))

    # Regress in log-sigma space (scheduler-agnostic) and schedule actual
    # forwards uniformly in log-sigma (works on karras / exponential / kl_optimal).
    log_sigma_min, log_sigma_max, delta_ls_unit = _resolve_schedule_stats(
        model_sampling, scheduler, steps, warmup_steps, stop_caching_step
    )

    forbid_consec = sampler_name in _FRAGILE_SAMPLERS
    if forbid_consec:
        logger.info(
            f"Spectrum: {sampler_name!r} is fragile (multistep / SDE / "
            "ancestral) — forbidding consecutive cached steps to avoid "
            "FD amplification or feedback compounding."
        )

    state = SpectrumState(
        window_size=window_size,
        flex_window=flex_window,
        warmup_steps=warmup_steps,
        w=blend_w,
        m=cheby_degree,
        lam=ridge_lambda,
        num_steps=steps,
        stop_caching_step=stop_caching_step,
        log_sigma_min=log_sigma_min,
        log_sigma_max=log_sigma_max,
        delta_ls_unit=delta_ls_unit,
        forbid_consecutive_cache=forbid_consec,
    )

    # Install capture hook once per DiT instance (no-op on subsequent runs) and
    # bind this sample's state to the module. The hook reads state from the
    # module attribute, so its identity/closure is stable across samples —
    # torch.compile's dynamo cache survives between runs.
    _ensure_capture_hook(dit)
    dit.final_layer._spectrum_state = state

    old_wrapper = m.model_options.get("model_function_wrapper")

    def spectrum_wrapper(apply_model, args):
        input_x = args["input"]
        timestep = args["timestep"]
        c = args["c"]
        cond_or_uncond = args["cond_or_uncond"]

        sigma_val = timestep[0].item()
        log_sigma = math.log(max(sigma_val, _LOG_SIGMA_EPS))

        if state.last_sigma is None or abs(sigma_val - state.last_sigma) > 1e-8:
            if state.step_idx >= 0:
                if state.mode == "actual":
                    state.fwd_count += 1
                    state.consec_cached = 0
                    state.note_actual(state.last_sigma_log)
                else:
                    state.consec_cached += 1

            state.step_idx += 1
            state.last_sigma = sigma_val
            state.last_sigma_log = log_sigma
            state.mode = "cached" if state.should_cache(log_sigma) else "actual"

        if state.mode == "cached" and state.has_forecasters(cond_or_uncond):
            predictions = []
            for cou in cond_or_uncond:
                pred_feat = state.forecasters[cou].predict(log_sigma)
                predictions.append(pred_feat)

            batched_feat = torch.cat(predictions, dim=0)
            t_internal = model_sampling.timestep(timestep).to(batched_feat.dtype)
            noise_pred = _spectrum_fast_forward(dit, t_internal, batched_feat)
            return model_sampling.calculate_denoised(
                timestep, noise_pred.float(), input_x
            )

        state.mode = "actual"

        if old_wrapper is not None:
            result = old_wrapper(apply_model, args)
        else:
            result = apply_model(input_x, timestep, **c)

        feat = state.captured_feat
        if feat is not None:
            batch_chunks = len(cond_or_uncond)
            feat_chunks = feat.chunk(batch_chunks, dim=0)
            for idx, cou in enumerate(cond_or_uncond):
                if cou not in state.forecasters:
                    state.forecasters[cou] = SpectrumPredictor(
                        state.m_param,
                        state.lam,
                        state.w,
                        feat.device,
                        feat_chunks[idx].shape,
                        log_sigma_min=state.log_sigma_min,
                        log_sigma_max=state.log_sigma_max,
                    )
                state.forecasters[cou].update(log_sigma, feat_chunks[idx])

        return result

    m.set_model_unet_function_wrapper(spectrum_wrapper)

    latent_img = latent_image["samples"].clone()
    latent_img = comfy.sample.fix_empty_latent_channels(
        m, latent_img, latent_image.get("downscale_ratio_spacial")
    )

    batch_inds = latent_image.get("batch_index")
    noise = comfy.sample.prepare_noise(latent_img, seed, batch_inds)

    noise_mask = latent_image.get("noise_mask")
    callback = latent_preview.prepare_callback(m, steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    try:
        samples = comfy.sample.sample(
            m,
            noise,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent_img,
            denoise=denoise,
            noise_mask=noise_mask,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=seed,
        )
    finally:
        dit.final_layer._spectrum_state = None
        if hasattr(dit, "_mod_pooled_proj"):
            del dit._mod_pooled_proj

    if state.step_idx >= 0:
        if state.mode == "actual":
            state.fwd_count += 1
        else:
            state.consec_cached += 1

    actual = state.fwd_count
    total = state.step_idx + 1
    speedup = total / max(1, actual)
    do_cfg = not math.isclose(cfg, 1.0)
    cfg_note = " (x2 for CFG)" if do_cfg else ""
    logger.info(
        f"Spectrum: {actual}/{total} actual forwards "
        f"({speedup:.2f}x theoretical speedup{cfg_note})"
    )

    out = latent_image.copy()
    out.pop("downscale_ratio_spacial", None)
    out["samples"] = samples
    return (out,)
