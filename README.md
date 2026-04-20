# Spectrum for ComfyUI

Training-free diffusion sampling acceleration via **Chebyshev polynomial feature forecasting** ([Han et al., CVPR 2026](https://arxiv.org/abs/2603.01623)). Drop-in KSampler replacement that skips transformer blocks on predicted steps for ~2-3x speedup.

## How it works

Standard diffusion runs the full DiT (all transformer blocks) at every denoising step. Spectrum observes that block outputs are smooth functions of **log-sigma**, so most steps can be **predicted** instead of computed.

On "actual" steps the full model runs and block outputs are captured. On "cached" steps all transformer blocks are skipped — only `t_embedder` + `final_layer` + `unpatchify` execute, using features predicted from a Chebyshev ridge-regression fit in log-sigma space.

### Log-sigma-uniform scheduling

Both the regression axis and the actual/cached decision operate on log-sigma, so Spectrum runs against any scheduler — `simple`, `karras`, `exponential`, `sgm_uniform`, `ddim_uniform`, `beta`, `kl_optimal`. **Quality varies sharply by scheduler though: `simple` is the validated default**, the rest carry residual drift even with all guards on. After `warmup_steps` seed samples, an actual forward runs whenever log-sigma has moved more than a threshold from the last actual; the threshold grows by `flex_window` after each actual (predictor trusted more as it accumulates samples). Steps beyond `stop_caching_step` always run actual forwards for final-detail refinement; on uniform-tail schedulers Spectrum auto-pushes that step earlier to compensate.

With 28 steps and defaults: ~**15 actual forwards** out of 28 total steps (~1.8× theoretical speedup).

## Usage

Place the **KSampler (Spectrum)** node where you'd normally use a KSampler. Same inputs as the stock KSampler (model, seed, steps, cfg, sampler, scheduler, conditioning, latent). Chains with other model wrappers (Flex Attention, Flash Attention 4, etc.).

### Samplers

#### Recommended: `er_sde` + `simple` (validated)

**This is the empirically-tested combination.** Use it unless you have a reason to deviate. Neutral style, flat colors, sharp lines, numerically stable under caching. The `simple` scheduler's huge late log-σ gaps act as a self-correcting "tail snap" — accumulated cache errors get pulled straight to the denoised in the final steps. Other scheduler shapes (karras / exponential / kl_optimal) lack this wash-out and produce visible drift at higher cache rates even with all guards on. Speedup capped around 1.5× by the no-consecutive-caches guard; quality matches stock `er_sde + simple` in our tests.

#### Other tested samplers

- **`euler_a`** — full speedup achievable on `simple`. Single-step, softer/thinner lines, CFG can be pushed higher. On non-`simple` schedulers it now triggers the consecutive-cache guard (added after the analytical drift bench showed 10× drift amplification on euler_a + karras vs euler_a + simple); use `simple` to get the full speedup back.
- **`dpmpp_2m_sde_gpu`** — guarded; ~1.5× cap. Style similar to `er_sde` with more prompt-driven variety. Stick to `simple` here too.

**Guarded-sampler set** (auto-detected, no user action needed): all multistep / SDE samplers (`er_sde`, `dpmpp_sde(_gpu)`, `dpmpp_2m(_cfg_pp)`, `dpmpp_2m_sde(_heun)(_gpu)`, `dpmpp_3m_sde(_gpu)`, `lms`, `ipndm`, `ipndm_v`, `deis`, `res_multistep*`, `sa_solver(_pece)`) plus all single-step ancestrals (`euler_ancestral(_cfg_pp)`, `dpm_2_ancestral`, `dpmpp_2s_ancestral(_cfg_pp)`).

#### Scheduler note

`simple` is the only validated scheduler. Front-loaded schedulers (`karras`, `exponential`, `kl_optimal`) work but introduce visible drift even with guards — the analytical drift bench (`bench/spectrum/analyze_drift.py` in the anima_lora repo) shows 4–9× more endpoint drift on these vs `simple` for the same sampler. Spectrum partially compensates by automatically pushing `stop_caching_step` earlier on uniform-tail schedules (see "Parameters → `stop_caching_step`").

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `window_size` | 2.0 | Initial log-sigma gap between actuals (in units of the schedule's per-step log-sigma move). Larger = more caches between actuals. |
| `flex_window` | 0.25 | Threshold growth per actual forward (same units). |
| `warmup_steps` | 7 | Steps that always run full forward (seeds the forecaster). |
| `blend_w` | 0.3 | Chebyshev/Taylor blend weight (1.0 = pure Chebyshev). |
| `cheby_degree` | 3 | Number of Chebyshev basis functions. |
| `ridge_lambda` | 0.1 | Ridge regression regularization strength. |
| `stop_caching_step` | -1 | Step at which caching stops; remaining steps always actual. `-1` = auto: picks based on the schedule's late log-σ tail-snap ratio. `simple` keeps `steps - 3` (full speedup); uniform-tail schedules (karras / exp / kl_optimal) push earlier (≈ `steps - 5..8`) to dilute accumulated drift. Override with a positive integer to force a specific value. |

### Tuning tips

- **More speedup**: increase `flex_window` (fewer actuals as denoising progresses).
- **Better quality**: increase `warmup_steps`, decrease `flex_window`, or decrease `stop_caching_step`.
- **Aggressive acceleration**: `flex_window=1.0`, `blend_w=0.7` on `euler_a + simple` (~3-4× speedup).
- **Front-loaded schedulers (karras / exponential / kl_optimal)**: log-sigma-uniform scheduling + the auto `stop_caching_step` heuristic handle them, but they still drift more than `simple`. Prefer `simple` if quality matters; use these only if your workflow already requires them.

## Modulation guidance

The **KSampler (Spectrum + Mod Guidance)** and **Advanced** variants add text-conditioned quality steering via a learned `pooled_text_proj` MLP adapter ([Starodubcev et al., ICLR 2026](https://arxiv.org/abs/2502.15349)). The adapter projects pooled text embeddings into a guidance delta that is injected into the DiT's AdaLN timestep embedding, steering generation toward the specified quality attributes.

The default ~12MB `pooled_text_proj` weight is auto-downloaded on first use from the [anima_lora release page](https://github.com/sorryhyun/anima_lora/releases/tag/mod_guidance) into `ComfyUI/models/anima_mod_guidance/`. The simple node always uses the default; the advanced node exposes an adapter dropdown where `(auto-download default)` triggers the same download or you can pick a custom adapter from `loras/`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `clip` | — | CLIP encoder for encoding quality tags |
| `adapter` | `(auto-download default)` | `pooled_text_proj` safetensors file (advanced node only) |
| `quality_tags` | `absurdres, highres, masterpiece, ...` | Quality/aesthetic tags to steer toward |
| `mod_w_profile` (simple) | `step_i8_skip27` | Per-block guidance preset. `step_i8_skip27` (default, best quality) protects blocks 0–7 + 27 and applies `w=3` to blocks 8–26. `step_i14` is the safe option — use it when a LoRA shows anatomy drift. `uniform_w3` recovers pre-0413 legacy behavior. |
| `mod_w` (advanced) | 3.0 | Peak guidance strength applied per-block |
| `mod_start_layer` (advanced) | 8 | First block (inclusive) that receives the steering delta. `0` = uniform legacy behavior |
| `mod_end_layer` (advanced) | -1 | Last block + 1 (exclusive). `-1` = all remaining blocks. Set to `27` to skip Anima's compensation block |
| `mod_taper` (advanced) | 0 | Number of late slots to scale by `mod_taper_scale`. `0` disables taper |
| `mod_taper_scale` (advanced) | 0.25 | Multiplier for tapered slots |
| `mod_final_w` (advanced) | 0.0 | `w` applied at `final_layer`. `0` = don't disturb the output head |

Per-block guidance schedules address quality drift on LoRAs whose distribution sits far from the positive-prompt axis (e.g. early blocks blowing out tonal DC into uniform color collapse). The default `step_i8_skip27` protects blocks 0–7 and the final compensation block 27 from the steering delta while keeping the base text projection uniform across all blocks. See `docs/mod-guidance.md` in the anima_lora repo for the underlying rationale.
