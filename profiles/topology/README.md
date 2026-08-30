# Topology-first profile view

This directory is an index, not an alternative schema. Profiles are still defined by
`profile.json` files under `profiles/<model>/<topology>/...`.

Use this view if you reason about a machine first and then pick a model profile.

- `single-r9700`: one R9700 topology (Muse V1 currently available)
- `dual-r9700`: dual R9700 topology (Qwen profile currently available)

To launch a profile by ID, still use the catalog:

```bash
./r9v list --by-topology
./r9v show <profile-id>
./r9v run <profile-id> --model-dir ...
```
