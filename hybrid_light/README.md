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
  --lights-json /home/majortom/project/IndoorLightEditing/examples/Example1/explicit_lights/latest/light_predictions.json \
  --mode local --spp 64
```

The renderer stores both `render_local_raw.exr` and an OptiX-denoised
`render_local.exr`/`render_local.png`. Pass `--no-denoise` for an untouched
Monte Carlo result.

ILE intensity is a relative initialization. Use `--radiance-scale` to calibrate
it before implementing per-light differentiable intensity refinement. If ILE
normalizes depth in a future run, export `camera.depth_normalized` and
`camera.depth_scale`, or pass the matching `--geometry-scale` explicitly.
