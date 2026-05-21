"""SPD (Spectral Progressive Diffusion) — multi-resolution inference, composed
with Spectrum as the "SPEED" sampler.

Xiao et al., arXiv:2605.18736. Grow spatial resolution along the denoising
trajectory: run early (noise-dominated) steps at low resolution, then inject
high-frequency detail via *spectral noise expansion* once finer frequencies
emerge from noise. The latent power spectrum decays as a power law on Anima
(β≈2.26), so HF carries little signal and is cheap to defer.

This is the **naive-reset compose** that the SPD∘Spectrum bench validated as the
ship path (`anima_lora/bench/spd/compose_report.md`, Phase 0(a)): run the
low-res prefix with no Spectrum caching, then at the σ handoff reset Spectrum's
forecaster and let it forecast over the full-res tail (phase-2-only). The
fancier band-aligned forecaster (Phase 0(b)) was *falsified* — do not build it.

Unlike Spectrum (a per-forward model wrapper) SPD must own the whole denoise
loop: it changes the latent grid mid-loop (`spectral_expand`) and re-spaces the
remaining σ schedule, neither of which a `model_function_wrapper` can do. So it
is implemented as a custom ``comfy.samplers.KSAMPLER`` sample function. Spectrum
stays the model wrapper underneath; this sampler reaches into the bound
``SpectrumState`` and resets it at the handoff.

The DCT primitives below are ported verbatim from ``anima_lora/networks/spd.py``
(``dct2`` / ``idct2`` / ``dct_lowpass_init`` / ``spectral_expand``) so the
spectral-expansion geometry matches the CLI sampler bit-for-bit.

v0 scope:
  * **Euler only** — spectral expansion re-spaces σ mid-loop, which precomputed
    multistep coefficients cannot follow (matches the CLI ``--spd`` constraint).
  * **Single late knee** — one low→full transition. ``spd_scale`` is the prefix
    resolution, ``spd_sigma`` the handoff σ. Validated point: 0.5 / 0.7.
  * **Phase-2-only Spectrum** — no caching during the low-res prefix.
"""

import logging
import math

import torch

logger = logging.getLogger(__name__)

# Anima DiT spatial patch size: latent H/W must be even (pixel mod-16). The
# low-res prefix grid and the expanded grid are both snapped to this multiple.
_PATCH = 2


# ── DCT helpers (2D separable, type-II, pure PyTorch) ──────────────────────────
# Ported verbatim from anima_lora/networks/spd.py to keep the spectral-expansion
# geometry identical to the CLI SPD sampler.


def _dct_matrix(n: int, device, dtype) -> torch.Tensor:
    nr = torch.arange(n, device=device, dtype=dtype)
    k = nr.unsqueeze(1)
    m = torch.cos(torch.pi * k * (2 * nr + 1) / (2 * n))
    m[0] *= 1.0 / math.sqrt(n)
    m[1:] *= math.sqrt(2.0 / n)
    return m


def dct2(x: torch.Tensor) -> torch.Tensor:
    """2D type-II DCT over the last two dims of a (B, C, H, W) tensor."""
    B, C, H, W = x.shape
    Dh = _dct_matrix(H, x.device, x.dtype)
    Dw = _dct_matrix(W, x.device, x.dtype)
    y = x.reshape(B * C, H, W)
    y = Dh @ y
    y = y @ Dw.T
    return y.reshape(B, C, H, W)


def idct2(x: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`dct2` (last two dims of a (B, C, H, W) tensor)."""
    B, C, H, W = x.shape
    Dh = _dct_matrix(H, x.device, x.dtype)
    Dw = _dct_matrix(W, x.device, x.dtype)
    y = x.reshape(B * C, H, W)
    y = Dh.T @ y
    y = y @ Dw
    return y.reshape(B, C, H, W)


def _snap(v: float, mult: int) -> int:
    """Round to nearest positive multiple of ``mult`` (DiT patch_spatial)."""
    return max(mult, int(round(v / mult)) * mult)


def dct_lowpass_init(x5: torch.Tensor, scale: float, patch: int) -> torch.Tensor:
    """DCT low-pass of a (B,C,1,H,W) latent down to a (B,C,1,h,w) grid (paper T_Φ)."""
    B, C, T, H, W = x5.shape
    x4 = x5.squeeze(2).float()
    xi = dct2(x4)
    h = min(_snap(H * scale, patch), H)
    w = min(_snap(W * scale, patch), W)
    x_low = idct2(xi[:, :, :h, :w])
    return x_low.unsqueeze(2).to(x5.dtype)


def spectral_expand(
    x5: torch.Tensor,
    sigma_val: float,
    scale_lo: float,
    scale_hi: float,
    H_full: int,
    W_full: int,
    patch: int,
    gen: torch.Generator,
    hf_scale: float = 1.0,
) -> tuple[torch.Tensor, float]:
    """Embed the current low-res DCT block into a larger grid, fill HF slots with
    σ-scaled noise, iDCT, scale by κ (Eq. iii) and align the timestep (Eq. 5–6).

    Returns (expanded (B,C,1,h_hi,w_hi) latent, sigma_aligned).
    """
    B, C, T, h_lo, w_lo = x5.shape
    x4 = x5.squeeze(2).float()
    xi = dct2(x4)

    h_hi = max(_snap(H_full * scale_hi, patch), h_lo)
    w_hi = max(_snap(W_full * scale_hi, patch), w_lo)

    r = scale_hi / scale_lo
    sigma_aligned = (r * sigma_val) / (1.0 + (r - 1.0) * sigma_val)
    kappa = r / (1.0 + (r - 1.0) * sigma_val)

    xi_new = torch.zeros(B, C, h_hi, w_hi, device=x5.device, dtype=torch.float32)
    xi_new[:, :, :h_lo, :w_lo] = xi
    noise = torch.randn(
        xi_new.shape, generator=gen, device=x5.device, dtype=torch.float32
    )
    mask = torch.zeros_like(xi_new)
    mask[:, :, h_lo:, :] = 1.0
    mask[:, :, :h_lo, w_lo:] = 1.0
    xi_new = xi_new + mask * sigma_val * noise * hf_scale

    x4_new = idct2(xi_new) * kappa
    return x4_new.unsqueeze(2).to(x5.dtype), float(sigma_aligned)


# ── SPEED sampler: custom KSAMPLER that owns the multi-resolution Euler loop ────


def make_speed_sampler(state, spd_scale: float, spd_sigma: float, seed: int):
    """Build a ``comfy.samplers.KSAMPLER`` running the SPD multi-resolution Euler
    loop with Spectrum caching layered on the full-res tail.

    ``state`` is the live :class:`spectrum.SpectrumState` already bound to the
    DiT's ``final_layer`` (so the Spectrum model wrapper is active underneath).
    The sampler runs the low-res prefix with ``state.active = False`` (all-actual,
    no forecaster), then at ``σ ≤ spd_sigma`` spectral-expands to full res,
    re-spaces the remaining σ schedule, calls ``state.reset()`` and flips
    ``state.active = True`` so Spectrum re-warms over the tail — the bench's
    "naive reset" compose.

    Euler-only by construction: the σ re-spacing happens on the schedule array
    this sampler iterates, so it cannot be expressed through a precomputed
    multistep coefficient table.
    """
    import comfy.samplers
    from comfy.k_diffusion.sampling import to_d
    from comfy.utils import model_trange as trange

    spd_scale = float(spd_scale)
    spd_sigma = float(spd_sigma)

    @torch.no_grad()
    def speed_sample(model, x, sigmas, extra_args=None, callback=None, disable=None):
        extra_args = {} if extra_args is None else extra_args
        s_in = x.new_ones([x.shape[0]])
        gen = torch.Generator(device=x.device).manual_seed(int(seed) + 10_000)
        H_full, W_full = int(x.shape[-2]), int(x.shape[-1])
        sigmas = sigmas.detach().clone().float()
        n = len(sigmas) - 1

        # The DCT helpers operate on a (B,C,1,H,W) view. Anima latents already
        # arrive 5D with T=1; only the bare 4D (B,C,H,W) case needs un/squeezing.
        nd = x.ndim
        to5 = lambda t: t.unsqueeze(2) if nd == 4 else t
        from5 = lambda t5: t5.squeeze(2) if nd == 4 else t5

        cur_scale = spd_scale
        transitioned = cur_scale >= 1.0
        if cur_scale < 1.0:
            x = from5(dct_lowpass_init(to5(x), cur_scale, _PATCH)).to(x.dtype)

        # Phase-2-only: Spectrum stays inactive (all-actual, no forecaster) until
        # the handoff. If there is no low-res prefix it is active from step 0.
        state.active = transitioned

        # The Euler body mirrors comfy.k_diffusion.sampling.sample_euler (s_churn=0)
        # op-for-op — same to_d, tensor sigma_hat / dt — so the no-transition path
        # is bit-for-bit identical to the stock Spectrum sampler (the R3 gate).
        for i in trange(n, disable=disable):
            # Resolution handoff: expand the low-res latent to full res, re-space
            # the remaining schedule (and σ at index i), and re-arm Spectrum for a
            # fresh full-res warmup.
            if (not transitioned) and float(sigmas[i]) <= spd_sigma:
                old = float(sigmas[i])
                x5, sigma_new = spectral_expand(
                    to5(x), old, cur_scale, 1.0, H_full, W_full, _PATCH, gen
                )
                x = from5(x5).to(x.dtype)
                if old > 0.0 and sigma_new != old:  # re-space remaining σ (Sec 4.3)
                    sigmas[i + 1:] = sigma_new * (sigmas[i + 1:] / old)
                sigmas[i] = sigma_new  # query the model at the aligned σ̃
                cur_scale = 1.0
                transitioned = True
                state.reset()
                state.active = True
                # reset() restarts step_idx at the tail's step 0, so the cache
                # logic must measure the horizon over the tail, not the full
                # schedule — otherwise should_cache's last-3-actual guard
                # (stop_at = num_steps - 3) never fires on a short tail and the
                # final full-res step gets forecasted instead of computed. This
                # also realigns the forecaster's τ normalization (total_steps)
                # with the reset step_idx.
                state.num_steps = n - i

            sigma_hat = sigmas[i]
            denoised = model(x, sigma_hat * s_in, **extra_args)
            d = to_d(x, sigma_hat, denoised)
            if callback is not None:
                callback(
                    {"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat,
                     "denoised": denoised}
                )
            dt = sigmas[i + 1] - sigma_hat
            x = x + d * dt

        return x

    return comfy.samplers.KSAMPLER(speed_sample)
