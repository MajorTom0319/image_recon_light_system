# IndoorLightEditing → Materialist prototype

This bridge loads the light meshes and intensities exported in
`light_predictions.json`, keeps their current camera-space convention
(`+X` right, `+Y` up, `-Z` forward), and attaches the meshes to a Materialist
scene as Mitsuba emitters. Materialist's reconstructed PLY and per-pixel
albedo/roughness/metallic/normal maps remain responsible for geometry and BRDF.

The fixed-light material optimizer includes visible/invisible lamps and
visible/invisible windows. Lamps use ordinary area emitters; each window keeps
its finite OBJ aperture and the original ILE sun/sky/ground directional
spherical-Gaussian lobes. Other hybrid entries remain lamp-only by default
because their Stage-B parameterization assumes one uniform RGB value per lamp.

Prepare a consistent Materialist scene and IndoorLightEditing depth first:

```bash
/home/majortom/miniconda3/envs/materialist5090/bin/python \
  test_matnet_infer_moge2.py \
  --image-path /home/majortom/project/IndoorLightEditing/examples/Example1/input/im.png \
  --output-dir output_imgs/indoorlightediting_test1 \
  --scale 1
```

This command treats PNG/JPG as sRGB, writes a linear `gt_image.exr`, preserves
an exact display-color `gt_image.png`, and rebuilds the mesh every run. It also
exports validity masks, camera-space and Materialist-space MoGe vectors,
`mesh_depth.exr`, and `inference_manifest.json`. `moge2_normal.exr` and
`moge2_points.exr` use Materialist's `+X right, +Y up, -Z forward` convention;
their `*_camera.exr` counterparts retain the raw MoGe camera convention.

Run a validation pass first:

```bash
/home/majortom/miniconda3/envs/materialist5090/bin/python \
  scripts/render_ile_lights.py \
  --materialist-dir output_imgs/indoorlightediting_test \
  --lights-json /home/majortom/project/IndoorLightEditing/examples/Example1/explicit_lights/latest/light_predictions.json \
  --dry-run
```

Inspect `hybrid_ile_render/lights_projected.png`, then render local lights:

```bash
/home/majortom/miniconda3/envs/materialist5090/bin/python \
  scripts/render_ile_lights.py \
  --materialist-dir output_imgs/indoorlightediting_test \
  --lights-json /home/majortom/project/IndoorLightEditing/examples/Example1/explicit_lights/20260806_152412/light_predictions.json \
  --mode local --spp 128
```

The renderer stores both `render_local_raw.exr` and an OptiX-denoised
`render_local.exr`/`render_local.png`. Pass `--no-denoise` for an untouched
Monte Carlo result.

ILE intensity is a relative initialization. Use `--radiance-scale` to calibrate
it before implementing per-light differentiable intensity refinement. If ILE
normalizes depth in a future run, export `camera.depth_normalized` and
`camera.depth_scale`, or pass the matching `--geometry-scale` explicitly.

## Stage B + Stage C optimization

The separate optimization entry freezes Materialist geometry/materials,
optimizes one scale per ILE lamp, then freezes those local lamps and optimizes a
32x16 far-field HDRI:

```bash
/home/majortom/miniconda3/envs/materialist5090/bin/python \
  scripts/optimize_ile_farfield.py \
  --materialist-dir output_imgs/indoorlightediting_test \
  --lights-json /home/majortom/project/IndoorLightEditing/examples/Example1/explicit_lights/20260806_152412/light_predictions.json \
  --stage-b-iters 200 \
  --farfield-iters 300 \
  --spp 64 --spp-grad 16 --preview-spp 128
```

Stage B constrains each emitter to
`optimized_rgb = ILE_RGB * scalar_scale`. Its loss combines display-space
Charbonnier, MSE, and a `log(scale)` prior. Stage C keeps those optimized lamps
fixed and optimizes an HxW tensor of `(16, 32, 3)` with non-negativity, energy,
and spherical horizontal-wrap TV regularization. Unless `--target` is passed,
this entry now uses the original image recorded by the ILE JSON instead of a
possibly stale legacy `gt_image.exr`.

The default output directory is `hybrid_ile_farfield_opt/`. Important outputs
include `stage_b_lamps.json`, both optimization histories, optimized local and
combined renders, `farfield_optimized_32x16.exr/.hdr`, and
`optimization_manifest.json`. Use `--stage-b-only` to stop before the HDRI
stage.

## Fixed local lights: Stage C + material refinement

For the experiment that disables Stage B, keeps all input ILE lamp and window
radiance fixed, optimizes the 32x16 far-field HDRI first, and then optimizes
Materialist ARM maps:

```bash
/home/majortom/miniconda3/envs/materialist5090/bin/python \
  scripts/optimize_ile_farfield_materials.py \
  --materialist-dir output_imgs/indoorlightediting_test \
  --lights-json /home/majortom/project/IndoorLightEditing/examples/Example1/explicit_lights/20260806_152412/light_predictions.json \
  --farfield-iters 1000 \
  --material-iters 1000 \
  --material-order rm a \
  --model_name pos_mlp \
  --integrator prb --max-depth 4 \
  --spp 16 --spp-grad 16 \
  --validation-spp 64 --validation-seeds 2 \
  --final-spp 256
```

`--model_name none` retains the original direct per-pixel optimization. To use
Materialist's positional MLP for both Stage C and Stage D, change it to:

```bash
--model_name pos_mlp
```

This entry automatically calls `load_ile_lights(..., include_windows=True)`;
no extra CLI switch is required. A window record keeps the three ILE lobes
`src`, `srcSky`, and `srcGrd`, each encoded as
`[RGB, direction_x, direction_y, direction_z, concentration]`. For a direction
`w` from the shaded point toward the sampled window aperture, the emitted
radiance is

```text
L(w) = sum_k RGB_k * exp(lambda_k * min(dot(direction_k, w) - 1, 0))
       k in {sun, sky, ground}
```

The OBJ still determines the finite aperture, distance, visibility, and soft
shadow. The directional formula is evaluated inside the custom
`ile_window` Mitsuba emitter; reducing the three lobes to a uniform summed RGB
would discard the predicted directions and concentrations. `--radiance-scale`
multiplies all lamp RGB values and all three window RGB values, while leaving
window directions and concentrations unchanged.

Window shapes explicitly use Mitsuba's `null` BSDF. Omitting `bsdf` does not
mean transparent: Mitsuba supplies an opaque default material, which appears as
a black aperture when the custom emitter has no camera-facing radiance. The
visible coplanarity offset is therefore applied only to visible lamps; windows
remain at their predicted OBJ positions so their apertures stay aligned with
the wall. The same rule is used by `lights_projected.png`.

The PosMLP branch parameterizes the HDRI with spherical coordinates and the
ARM maps with image-space UV coordinates. It keeps the same loss terms,
constraints, `rm -> a` phase order, best-checkpoint selection, frozen-channel
checks, HDRI reload, and final validation gate as the direct branch. Its HDRI
head is initialized to `--farfield-init`, rather than the PosMLP default
`softplus(0)`. Both modes export the same EXR/HDR/PNG files; the selected mode
is recorded as `model_name` in `optimization_manifest.json`. The default is
`none`, so existing commands retain their previous behavior. PosMLP requires
CUDA because gradients pass between PyTorch and Mitsuba/Dr.Jit.

An explicit `--target` always has highest priority. Otherwise a completed
inference manifest with schema v2 or newer selects the corrected linear
`gt_image.exr`, which shares the material/mesh working resolution. Legacy runs
without a verified manifest fall back to the original image recorded in the
ILE JSON, avoiding old EXRs that stored sRGB values as linear. `target.png` is
only a preview and never enters the loss.

The default material order is `rm -> a`: roughness and metallic are optimized
together first, their best checkpoint is frozen, and albedo is optimized last.
Each phase gets a fresh optimizer and StepLR-equivalent schedule. Direct pixels
use Mitsuba Adam with `amsgrad=True` and `mask_updates=True`; PosMLP uses
Adam/AdamW with an explicit zero weight decay. The loss follows the real-image
branch in `inverse_img_w_mi_ori.py`: clipped standard-sRGB adaptive MSE + L1 and
an L1 prior to the input material maps. Clipping linear radiance to `[0, 1]`
before the standard transfer function prevents HDR fireflies from dominating
an LDR target. Default learning rate (`3e-4`), prior weight (`0.1`) and material
bounds remain unchanged. Detached mean-exposure matching is available via
`--material-exposure-match`, but is disabled by default. Albedo and metallic
are clamped to `[0, 1]`; roughness is clamped to `[0.07, 1]`.

Normal is fixed rather than optimized. The default is the loaded MoGe2 normal
map, matching the mesh reconstructed from MoGe2 depth. Pass
`--use-mesh-normal` only for the interpolated PLY-normal ablation.

Before optimization, the script writes `farfield_loss_mask.png` and
`material_loss_mask.png`. Visible lamp/window masks are thresholded and
dilated; depth discontinuities, validity boundaries and their neighborhoods
are also excluded. The material mask additionally requires valid, eroded mesh
pixels. Mask paths and retained fractions are recorded under `loss_masks` in
`optimization_manifest.json`.

Stage C has one authoritative output: `farfield_optimized_32x16.exr/.hdr`.
After exporting it, Stage D rebuilds its Mitsuba scene directly from that EXR;
the manifest records both the reload difference and the change during material
optimization, which must remain zero. `farfield_optimized_combined.*` is the
combined rendering produced by this same best HDRI checkpoint. The auto-exposed
`farfield_optimized_32x16_preview.png` is only for inspecting low-radiance HDRI
structure and is never used for rendering.

Stage C and every material phase use periodic fixed multi-seed validation. The
defaults evaluate two seeds at 64 SPP every 25 iterations and stop after eight
validation checks without the required relative improvement. Checkpoint
selection uses masked validation display MSE rather than a single noisy
training sample. If the Stage-C candidate has a worse display MSE than the
initial HDRI, the initial HDRI is restored before export and Stage D; the
rejected candidate remains available as `farfield_candidate_combined.*`. The
compatibility alias `--preview-spp` now maps to `--validation-spp`. Inspect
`material_phase_summaries.json` for the actual order, per-phase best MSE, and
frozen-channel error; a nonzero error above tolerance aborts the run. Also
inspect `farfield_metrics.json`, `final_metrics.json`, and the corresponding
`*_target_render_error.png` comparisons. `optimization_manifest.json` records
`fixed_local_lights`, `fixed_window_count`, and `fixed_local_radiance_scale`, so
a run can be checked for all expected local lights. The current default output
directory is `hybrid_ile_windows_nullbsdf_farfield_material_opt_posmlp_1/`;
pass an explicit `--output-dir` when comparing `none` and `pos_mlp` to avoid
overwriting results.

Validation renders and final renders use separate budgets. The default final
render uses one independent seed at 256 SPP. `best_results/rendered_img.exr`
and `rendered_img_final_raw.exr` are authoritative raw linear results;
`rendered_img_final_denoised.exr/.png` are optional display previews and never
replace the raw result or its metrics.

The Example1 export resolves to four fixed local lights: one visible lamp, one
invisible lamp, one visible window, and one invisible window. Both direct and
PosMLP branches completed an end-to-end PRB smoke test with `max_depth=4`; the
generated masks retained about 87.8% of the 320×240 pixels. The full unit suite
contains 26 passing tests, including window offset/null-BSDF, clipped LDR loss,
masked metrics, depth jumps and valid/invalid mesh boundaries. The Python
emitter is JIT-compiled by Dr.Jit; the first CUDA use can take several minutes,
while later runs can reuse the cache. The current emitter uniformly samples the
finite window aperture rather than importance sampling a sharp sun SG, so
high-concentration windows may still need a higher final SPP.
