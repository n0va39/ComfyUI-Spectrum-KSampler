"""Spectrum: Adaptive Spectral Feature Forecasting for ComfyUI.

Drop-in KSampler replacement that accelerates diffusion sampling via
Chebyshev polynomial feature forecasting (Han et al., CVPR 2026).

On "actual" steps the full model runs and block outputs are captured.
On "cached" steps all transformer blocks are skipped — only t_embedder +
final_layer + unpatchify execute, using predicted features from a
Chebyshev ridge-regression fit. Works with any ComfyUI sampler (Euler,
DPM, er_sde, etc.) because caching is handled transparently inside the
model_function_wrapper.

Node tiers:
  - SpectrumKSampler: basic drop-in, sensible defaults
  - SpectrumKSamplerModGuidance: + modulation guidance (adapter, quality tags, w)
  - SpectrumKSamplerAdvanced: + full Spectrum tuning + modulation guidance
  - SpectrumSPDKSampler (SPEED): + SPD multi-resolution prefix (low-res early,
    spectral-expand to full res at the handoff, Spectrum-forecasted tail)
  - AnimaModGuidance: standalone mod-guidance model patcher (composes with any sampler)
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
