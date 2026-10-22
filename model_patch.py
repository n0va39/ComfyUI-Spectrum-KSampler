"""Sampling-scoped Spectrum for native models with explicit feature/head adapters.

Adapters own only model identification, feature capture and the native output
head. Forecast math and cache policy remain the existing Spectrum machinery.
"""
import logging
from contextvars import ContextVar

import torch

from .feature_forecast import TokenSpectrumState, _update_forecasters
from .spectrum import (
    _advance_spectrum_state, _can_use_cached_prediction,
    _mark_spectrum_patch_consumed, _maybe_rearm_for_new_execution,
    _normalize_cond_or_uncond, _resolve_live_components, _spectrum_batch_keys,
)

logger = logging.getLogger(__name__)
_NATIVE_CAPTURE = ContextVar("spectrum_native_feature_capture", default=None)


def record_feature(module, tensor):
    """Neutral module hook: retain tensors only within this forward's scope."""
    capture = _NATIVE_CAPTURE.get()
    if capture is not None and capture[0] is module:
        _, state, extract = capture
        state.features.append(extract(tensor).detach().clone())


def _capture_safe(adapter, dit, input_x, c, apply_model):
    from comfy.patcher_extension import WrappersMP, get_all_wrappers

    options = c.get("transformer_options", {})
    patcher = getattr(getattr(apply_model, "__self__", None), "current_patcher", None)
    return (
        c.get("control") is None
        and not get_all_wrappers(WrappersMP.APPLY_MODEL, options)
        and not get_all_wrappers(WrappersMP.DIFFUSION_MODEL, options)
        # Scheduled weight hooks can change weights without changing model ID.
        and not getattr(patcher, "hook_patches", None)
        and not getattr(patcher, "current_hooks", None)
        and adapter.capture_safe(dit, input_x, c)
    )


def apply_spectrum_model_patch(
    model, adapter, *, steps=30, window_size=2.0, flex_window=0.25,
    warmup_steps=6, tail_actual_steps=3, blend_w=0.3, cheby_degree=3,
    ridge_lambda=0.1, history_size=100, enabled=True, one_sampler_only=False,
    verbose=False, compat_policy="conservative",
):
    if not enabled:
        return model
    if history_size < cheby_degree + 2:
        raise RuntimeError("Native Spectrum requires history_size >= cheby_degree + 2.")
    dit, sampling = adapter.require_model(model)
    patched = model.clone()
    configured = TokenSpectrumState(
        window_size=window_size, flex_window=flex_window, warmup_steps=warmup_steps,
        w=blend_w, m=cheby_degree, lam=ridge_lambda, num_steps=steps,
        tail_actual_steps=tail_actual_steps, history_size=history_size,
        verbose=verbose, one_sampler_only=one_sampler_only, compat_policy=compat_policy,
    )
    native_scope = ContextVar(adapter.key, default=None)
    adapter.capture_owner(dit)
    _install_sampling_scope(patched, dit, configured, native_scope, adapter.key)
    old_wrapper = patched.model_options.get("model_function_wrapper")
    # Do not capture the patcher: it would form a reference cycle.
    model_options = patched.model_options

    def actual_forward(apply_model, args):
        if old_wrapper is not None:
            return old_wrapper(apply_model, args)
        return apply_model(args["input"], args["timestep"], **args["c"])

    def spectrum_wrapper(apply_model, args):
        input_x, timestep, c = args["input"], args["timestep"], args["c"]
        state = native_scope.get()
        # Direct model calls do not identify a new crop or sampling invocation.
        if state is None:
            return actual_forward(apply_model, args)
        live_dit, live_sampling = _resolve_live_components(apply_model, dit, sampling, state)
        _maybe_rearm_for_new_execution(state)
        if state.native_live_id != id(live_dit):
            state.reset()
            state.native_live_id = id(live_dit)
        owner = adapter.capture_owner(live_dit)
        new_step = _advance_spectrum_state(state, timestep.flatten()[0].item())
        if not _capture_safe(adapter, live_dit, input_x, c, apply_model):
            state.clear_forecasters()
            state.mode = "actual"
            _mark_spectrum_patch_consumed(state)
            return actual_forward(apply_model, args)

        branches, valid_chunks = _normalize_cond_or_uncond(args, input_x.shape[0])
        keys = _spectrum_batch_keys(c, branches)
        live_patcher = getattr(getattr(apply_model, "__self__", None), "current_patcher", None)
        live_options = getattr(live_patcher, "model_options", model_options)
        if new_step:
            state.mode = "cached" if valid_chunks and state.should_cache(keys) else "actual"
            if state.verbose:
                logger.info("Spectrum for %s: step %d/%d sigma=%.6g mode=%s",
                            adapter.name, state.step_idx + 1, state.num_steps,
                            state.last_sigma, state.mode)
        if _can_use_cached_prediction(
            state, keys, input_x, timestep, c, old_wrapper, live_options, args, valid_chunks
        ):
            feat = torch.cat([state.forecasters[k].predict(float(state.step_idx)) for k in keys])
            out = adapter.head(live_dit, live_sampling.timestep(timestep).float(), feat, input_x)
            _mark_spectrum_patch_consumed(state)
            return live_sampling.calculate_denoised(timestep, out.float(), input_x)

        state.mode = "actual"
        state.features = []
        token = _NATIVE_CAPTURE.set((
            owner, state, lambda tensor: adapter.extract_features(tensor, input_x, c),
        ))
        try:
            result = actual_forward(apply_model, args)
            features = state.features
        except Exception:
            state.clear_forecasters()
            raise
        finally:
            state.features = None
            _NATIVE_CAPTURE.reset(token)
        if state.active and valid_chunks and len(features) == 1 and features[0].shape[0] == input_x.shape[0]:
            _update_forecasters(state, features[0], input_x, keys)
        else:
            state.clear_forecasters()
            if state.verbose:
                logger.warning("Spectrum for %s: invalid feature capture; no forecast update", adapter.name)
        _mark_spectrum_patch_consumed(state)
        return result

    patched.set_model_unet_function_wrapper(spectrum_wrapper)
    return patched


def _install_sampling_scope(model, dit, configured, scope, key):
    from comfy.patcher_extension import WrappersMP, add_wrapper_with_key

    def sampling_wrapper(executor, noise, latent_image, sampler, sigmas, *args, **kwargs):
        _maybe_rearm_for_new_execution(configured)
        state = None
        if not configured.patch_consumed and len(sigmas) > 1:
            state = TokenSpectrumState(
                window_size=configured.window_size, flex_window=configured.flex_window,
                warmup_steps=configured.warmup_steps, w=configured.w, m=configured.m_param,
                lam=configured.lam, num_steps=len(sigmas) - 1,
                tail_actual_steps=configured.tail_actual_steps, history_size=configured.history_size,
                verbose=configured.verbose, one_sampler_only=False, compat_policy=configured.compat_policy,
            )
            state.native_live_id = id(dit)
            if configured.one_sampler_only:
                configured.patch_consumed = True
                configured.steps_seen += 1
        token = scope.set(state)
        try:
            return executor(noise, latent_image, sampler, sigmas, *args, **kwargs)
        finally:
            if state is not None:
                state.clear_forecasters()
                state.features = None
            scope.reset(token)

    add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, key, sampling_wrapper,
                         model.model_options, is_model_options=True)
