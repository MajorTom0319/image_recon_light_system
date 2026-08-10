# IndoorLightEditing → Materialist prototype

This bridge loads the lamp meshes and RGB intensities exported in
`light_predictions.json`, keeps their current camera-space convention
(`+X` right, `+Y` up, `-Z` forward), and attaches the meshes to a Materialist
scene as Mitsuba area emitters. Materialist's reconstructed PLY and per-pixel
albedo/roughness/metallic/normal maps remain responsible for geometry and BRDF.

The first version supports visible and invisible **lamps**. ILE windows are
reported but deliberately ignored because their sun/sky/ground lobes cannot be
faithfully represented by a uniform area emitter.

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
and spherical horizontal-wrap TV regularization.

The default output directory is `hybrid_ile_farfield_opt/`. Important outputs
include `stage_b_lamps.json`, both optimization histories, optimized local and
combined renders, `farfield_optimized_32x16.exr/.hdr`, and
`optimization_manifest.json`. Use `--stage-b-only` to stop before the HDRI
stage.

## Fixed lamps: Stage C + material refinement

For the experiment that disables Stage B, keeps the input ILE lamp radiance
fixed, optimizes the 32x16 far-field HDRI first, and then optimizes Materialist
ARM maps:

```bash
/home/majortom/miniconda3/envs/materialist5090/bin/python \
  scripts/optimize_ile_farfield_materials.py \
  --materialist-dir output_imgs/indoorlightediting_test \
  --lights-json /home/majortom/project/IndoorLightEditing/examples/Example1/explicit_lights/20260806_152412/light_predictions.json \
  --farfield-iters 300 \
  --material-iters 500 \
  --material-order rm a \
  --spp 64 --spp-grad 64 --preview-spp 512
```

By default, the target is the original image path recorded in the ILE JSON, not
the legacy `gt_image.exr`. This avoids treating sRGB PNG values stored in an old
EXR as linear data; `target.png` is only a preview and never enters the loss.

The default material order is `rm -> a`: roughness and metallic are optimized
together first, their best checkpoint is frozen, and albedo is optimized last.
Each phase gets a fresh Adam/StepLR and early-stopping state. The loss follows
the direct real-image branch in `inverse_img_w_mi_ori.py`: display-space
adaptive MSE + L1 and an L1 prior to the input material maps. Default learning
rate (`3e-4`), prior weight (`0.1`), bounds, and phase-specific improvement
thresholds also follow that branch. Detached mean-exposure matching remains
available via `--material-exposure-match`, but is disabled by default so the
saved physical render and optimized brightness use the same objective. Albedo
and metallic are clamped to `[0, 1]`; roughness is clamped to `[0.07, 1]`.
Mesh normal is fixed in this experiment.

Stage C has one authoritative output: `farfield_optimized_32x16.exr/.hdr`.
After exporting it, Stage D rebuilds its Mitsuba scene directly from that EXR;
the manifest records both the reload difference and the change during material
optimization, which must remain zero. `farfield_optimized_combined.*` is the
combined rendering produced by this same best HDRI checkpoint. The auto-exposed
`farfield_optimized_32x16_preview.png` is only for inspecting low-radiance HDRI
structure and is never used for rendering.

The material candidate still uses a same-seed high-SPP validation gate. Inspect
`material_phase_summaries.json` for the actual order, per-phase best MSE, and
frozen-channel error; a nonzero error above tolerance aborts the run. Also
inspect `farfield_metrics.json`, `final_metrics.json`, and the corresponding
`*_target_render_error.png` comparisons. The latest default output directory is
`hybrid_ile_farfield_material_opt_v5/`; older directories contain previous
material-order or HDRI-selection semantics and should not be mixed with v5.
