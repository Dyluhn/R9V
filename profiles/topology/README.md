# Topology-first profile view

This directory is an index, not an alternative schema. Profiles are still defined by
`profile.json` files under `profiles/<model>/<topology>/...`.

Use this view if you reason about a machine first and then pick a model profile.

- `single-r9700`: one R9700 topology (Muse V1 currently available)
- `dual-r9700`: dual R9700 topology (Qwen profile currently available)

Resolve a profile by ID through the catalog:

```bash
./r9v list --by-topology
./r9v show <profile-id>
```

Then follow that profile's linked installation/status page. Lifecycle commands
remain fail-closed when a required artifact or runnable runtime stage has not
been published.
