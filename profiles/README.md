# Adding an exact R9V setup

An R9V profile is a reproducible composition, not a loose tuning preset.

1. Add an immutable `packages/models/<model>/<quant>/package.json` with every
   required file, byte count, SHA256, immediate source repository, revision,
   and model license.
2. Add or reuse a `runtimes/<runtime>/runtime.json`. It must enumerate the GPU
   architecture, accepted qtypes/shapes, API surface, TP sizes, and safety
   invariants.
3. Add a `hardware/<profile>/hardware.json` with the topology that affects
   placement or synchronization.
4. Put route-derived hot/cold choices under `packages/placements/`; do not bake
   workload-specific rankings into the model package.
5. Compose the pieces in `profiles/<model>/<variant>/profile.json` and provide
   `doctor`, `fetch`, `build`, `run`, and `verify` commands.
6. Publish a qualification report with correctness, PP, TG, context, hardware,
   cold-route behavior, and every known limitation.

The lifecycle is:

`experimental` → `release-candidate` → `qualified` → `retired`

A profile becomes `qualified` only after the advertised download-to-user path
works end-to-end. Kernel microbenchmarks alone are not sufficient.

## Compatibility rules

- New quant or component precision: new model-package ID.
- New runtime ABI or incompatible source graph: new runtime ID.
- Same bytes on a different topology: new hardware/placement/profile, not a
  renamed model package.
- Profile selection never moves submodule HEADs.
- Non-default images, containers, and caches should be namespaced by the
  resolved profile hash.
- CLI/host environment overrides profile defaults; descriptors themselves are
  declarative JSON and are never sourced as shell.

Validate the catalog without touching a GPU:

```bash
./r9v validate
./r9v show <profile>
./r9v doctor <profile>
```

## Catalog layouts

The authoritative identity space is `model / topology` (for example
`qwen38-flash-next/dual-r9700/`), because that aligns with model package + runtime
compatibility and avoids identity collisions.

A topology-first view is also available under `profiles/topology/<topology>/` for
human navigation:

- `profiles/topology/single-r9700/`
- `profiles/topology/dual-r9700/`

Each topology README links to the same profile IDs and avoids duplicating
descriptors.

## Topology guidance

- If you are choosing by hardware shape first (single GPU vs dual GPU), use
  `profiles/topology/...` and the new `./r9v list --by-topology` mode.
- If you are choosing by model/quant first, use `profiles/<model>/<topology>/`.
- Use the descriptor composition (`profile.json`) as the source of truth for
  launch behavior, not the folder path alone.
