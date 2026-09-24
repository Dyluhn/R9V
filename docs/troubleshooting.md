# Troubleshooting Qwen3.8 Flash Next

Use the profile alias that matches the installed artifacts: `qwen38-mtp4` for
UD-IQ4_XS MTP4 or `qwen38-q4-xl` for UD-Q4_K_XL MTP4. Start with a read-only
check:

```bash
./r9v show qwen38-mtp4
./r9v doctor qwen38-mtp4 -- --host-only
```

For a running server, inspect the runtime with `./r9v doctor PROFILE -- --runtime`.
A failed check includes the observed value and corrective action. Keep the same
`--state-dir` for setup, start, doctor and support so evidence refers to one
profile state.

| Check or symptom | Action |
|---|---|
| Package download or hash failure | Check disk and `hf` access, then repair or re-fetch the exact profile package. Never launch after a failed hash. |
| GPU count, architecture, or BDF mismatch | Run `amd-smi list`; rerun setup with `--gpu-bdfs BDF0,BDF1` in the intended rank order. |
| Image ID is not preserved after `docker load` | Check `docker info -f '{{ .DriverStatus }}'`; it should identify `io.containerd.snapshotter.v1`. Docker 29 upgraded installations may still use the legacy store. Follow the official [containerd image-store guide](https://docs.docker.com/engine/storage/containerd/) and [daemon configuration reference](https://docs.docker.com/engine/daemon/). |
| Rootless Docker is in use | Follow Docker's [rootless mode guide](https://docs.docker.com/engine/security/rootless/), confirm the selected context/socket with `docker info`, and verify the containerd image-store check above. |
| Normal-zone pressure warning | Free host memory or reduce CPU-offloaded residency. Swap does not satisfy pinned-RAM requirements. |
| Requested headroom shortfall | Preserve every per-rank shortfall. Change the target deliberately or use the release seed to plan it; the resulting placement still needs local workload qualification. |
| Startup or JIT timeout | Start waits 900 seconds by default, 2,400 for `qwen38-mtp4-uncensored`. Increase `--timeout`, inspect `docker logs --tail 200 r9v-qwen38-flash-next`, and check image, model, PLE, and cache space. |
| Existing container blocks startup | Inspect the exact named container, save diagnostics, then deliberately stop/remove that container before retrying. |
| Runtime worker or transport failure | Keep the container running long enough to collect startup evidence; run `./r9v doctor PROFILE -- --runtime` and inspect worker identity and HIP-visible BDFs. |
| Support request | Run `./r9v doctor PROFILE --state-dir DIR` first, then `./r9v support PROFILE --state-dir DIR`; treat the bundle as private and do not publish prompts, completions, raw token IDs, or logs. |
| `disk-space` | Names each filesystem that is short, what still has to be written there (package, PLE table, image, compile cache) and how much is free. Free space there, or pick other directories with `--model-dir`, `--data-dir`, `--ple-path` or `R9V_CACHE_DIR`. |
| `vram-other-processes` | Names the processes holding VRAM on the R9V GPUs. A warning while enough stays free; a failure when what is left is below what R9V needs. Close those apps (including another model server) before start. |
| `api-exposure` | The API has no authentication. A warning means `R9V_HOST_BIND` publishes it beyond this machine; set it to `127.0.0.1` and rerun setup unless authentication sits in front of the port. |
| `runtime-overlays` | A file the runtime mounts over its image does not match its pinned SHA-256. Restore the checkout (`git status`, then `git checkout -- runtimes/`); start refuses until every file matches. |
| `ced-projector` | The CED projector is missing, differs from the hash the model package pins, or is for another split. Rerun setup with the same `--ced` mode to fetch it (`--ced quality` has its own projector), or start with `--ced off`. |
| `expert-limit-consistency` | A full mutable expert cache runs only its pinned placement: the expert ceilings, cache slots and cache ranks must be the profile's values. Restore them (setup saves them from `profile.env`). |
| PLE size or residency failure | Verify the payload is exactly 28,800,138,240 bytes, on the intended SSD, and re-run setup with `--ple-path` if reusing it. |

A host-only pass proves prerequisites only. Health proves readiness only. The
bounded workload qualification is required for each new placement and headroom
target. See [installation](installation.md) for the complete setup flow and
[LLM setup guide](../LLM_SETUP_GUIDE.md) for agent-operated installs.
