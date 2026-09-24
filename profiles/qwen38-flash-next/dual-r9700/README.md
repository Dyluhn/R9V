# Qwen3.8 Flash Next dual-R9700 configuration

This is the machine-configuration reference for the published Qwen3.8 Flash
Next R9V profile. The general installation sequence is in
[`docs/installation.md`](../../../docs/installation.md); this page explains
what each host-facing setting means, how to discover the correct value, and
how to respond to every class of doctor result.

The published profile is optimized for two 32 GiB Radeon AI PRO R9700 GPUs.
The GPUs do not need to occupy the same PCIe slots as the reference machine.
The rank order, PCIe path capacity, RAM policy, PLE storage, and expert
cache are explicit so similar machines can use the defaults while different
machines fail or warn with a concrete correction.

## Configuration flow

1. Fetch/arrange the model and derive the PLE file as described in the
   installation guide.
2. Copy the user configuration template outside the checkout.
3. Set the model, PLE, GPU-order, and optional policy values.
4. Run the static doctor. Resolve every `FAIL` and understand every `WARN`.
5. Launch. The launcher repeats static preflight automatically.
6. After the server handles a request, run the runtime doctor to prove that
   the optimized decode path and MTP actually executed.

```bash
cp profiles/qwen38-flash-next/dual-r9700/user-config.example.env \
  /path/to/my-r9v-qwen.env

export R9V_CONFIG_FILE=/path/to/my-r9v-qwen.env
export R9V_MODEL_DIR=/path/to/qwen38-r9v

./r9v doctor qwen38 --model-dir "$R9V_MODEL_DIR"
./r9v run qwen38 --model-dir "$R9V_MODEL_DIR"
./r9v doctor qwen38 --model-dir "$R9V_MODEL_DIR" --runtime
```

Configuration precedence is:

1. A value explicitly exported in the caller's shell.
2. A conditional value in `R9V_CONFIG_FILE`.
3. The published defaults in [`profile.env`](profile.env).

Keep the template's `: "${NAME:=value}"` form. It fills an unset value but
does not replace a value explicitly exported for one launch.

## Understanding doctor status

| Status | Meaning | Launch behavior |
|---|---|---|
| `PASS` | The detected state satisfies the selected profile. | Continues |
| `WARN` | The state is usable but unverified, incomplete, or needs conscious review. | Continues unless strict mode is enabled |
| `FAIL` | The state is incompatible, unsafe, or internally inconsistent. | Stops |
| `NOTE` | Context that prevents a common misinterpretation. | No effect |

Every `WARN` and `FAIL` prints a `FIX:` line. JSON output stores the same text
in the check's `remediation` field:

```bash
./r9v doctor qwen38 --model-dir "$R9V_MODEL_DIR" --json \
  > r9v-qwen-doctor.json
```

Set `R9V_DOCTOR_STRICT=1` when qualifying a new machine. In strict mode a
warning also returns a nonzero status. Normal launch uses strict mode `0` so
discovery-only warnings, such as an unlocked BDF order, can be corrected
without pretending the machine is broken.

## GPU identity and tensor-parallel rank order

### `R9V_VISIBLE_DEVICES`

This is the comma-separated HIP device order. The first value becomes TP rank
0 and the second becomes TP rank 1. It is not merely a visibility filter: the
published expert placement is asymmetric, so reversing the devices changes
which physical card receives the larger static placement and dynamic cache.

Numeric HIP indices follow KFD GPU-node order. `amd-smi` has a separate index
space and may list the same physical devices in a different order on mixed-GPU
or multi-switch systems. `R9V_VISIBLE_DEVICES` accepts numeric indices only;
AMD-SMI UUIDs must not be passed as ROCR UUIDs. Use the BDF lock to confirm
the physical order. Discover both views with:

```bash
amd-smi list
for node in /sys/class/kfd/kfd/topology/nodes/*; do
  test -r "$node/properties" || continue
  printf 'KFD node %s: ' "${node##*/}"
  grep -E '^(location_id|domain|gfx_target_version) ' "$node/properties"
done
```

The doctor resolves each configured HIP index through KFD to its physical BDF
and reports the corresponding `amd-smi` index. Trust that resolved mapping, not
an assumption that HIP index 0 must mean `amd-smi` GPU 0.

Example on a host where the two index spaces happen to agree:

```text
GPU: 0
    BDF: 0000:03:00.0
GPU: 1
    BDF: 0000:13:00.0
```

To keep that order:

```bash
: "${R9V_VISIBLE_DEVICES:=0,1}"
```

To reverse it:

```bash
: "${R9V_VISIBLE_DEVICES:=1,0}"
```

Do not reverse the devices simply because one is the display card. Choose the
order together with the expert placement/cache policy and then measure it.

### `R9V_EXPECTED_GPU_BDFS`

This locks physical PCI addresses in TP-rank order. It catches device-index
renumbering after a driver, BIOS, cabling, or hardware change.

For the example above:

```bash
: "${R9V_EXPECTED_GPU_BDFS:=0000:03:00.0,0000:13:00.0}"
```

If it is empty, the doctor reports a warning containing the exact detected
assignment to paste into the configuration file. Confirm that the order is
intentional before accepting it. If a BDF check fails, either reorder
`R9V_VISIBLE_DEVICES` or correct the lock; do not edit the expected BDF merely
to silence the failure.

### `R9V_EXPECTED_PCIE_LINKS`

This optional lock records the exact end-to-root path-capacity bottleneck
expected for each TP rank. It accepts a readable generation form or a
numeric transfer-rate form:

```bash
: "${R9V_EXPECTED_PCIE_LINKS:=Gen5x16,Gen4x4}"
# Equivalent:
: "${R9V_EXPECTED_PCIE_LINKS:=32x16,16x4}"
```

The values follow `R9V_VISIBLE_DEVICES` rank order. Leave the setting empty on
the first doctor run. If sysfs exposes both links, doctor prints the exact line
to paste into the configuration. Once locked, a capacity or width change fails
preflight instead of silently changing the performance profile.

R9V cannot set a physical PCIe generation or lane width. Those are determined
by the GPU, motherboard slot, BIOS lane allocation, riser/cable, and competing
devices. This setting only verifies the result. If it fails, repair the
hardware/firmware topology or deliberately update the lock after deciding the
new topology is correct.

The lock and bandwidth floor both walk the complete device-to-root path. They
use each hop's maximum speed and negotiated width for stable capacity because
an idle PCIe link may temporarily downshift speed while lane allocation and
degraded training remain significant. The report includes current state
as a diagnostic but never mistakes the endpoint alone for the full topology.
The negotiated width must be readable and positive at every link-bearing hop;
the device's maximum width cannot substitute for unknown slot allocation.
An incomplete path fails validation and produces no suggested PCIe lock.
The capacity estimate is an upper bound, not proof of the negotiated speed
under load: firmware speed caps or persistent downtraining can keep the
current speed below the advertised maximum. Compare current and maximum
values during the affected workload before interpreting a generation mismatch.

### `R9V_MIN_PCIE_BANDWIDTH_GBPS`

This is the minimum theoretical one-direction PCIe payload bandwidth for each
TP rank, not a benchmark result. The doctor reads maximum speed and negotiated width
of every hop from the device up to the root port, accounts for PCIe
link encoding, and applies the floor to the slowest hop. An endpoint that
negotiates x16 behind a x4 upstream bridge is therefore scored at the x4
bottleneck, and the report names the capping hop. Endpoint sysfs alone cannot
be trusted for this: a card can read Gen5x16 at the endpoint while an
upstream switch or bifurcated link caps the real path.

The published defaults are:

```bash
: "${R9V_MIN_PCIE_BANDWIDTH_GBPS:=0,0}"
```

Hard performance floors are disabled by default. The separate 15,7 GB/s
reference targets still warn on slower paths. Set nonzero hard floors only
when you want to reject a slower topology. The exact link lock independently
checks that the intended slot paths have the expected capacity. Inspect
a card manually with:

```bash
bdf=0000:03:00.0
cat "/sys/bus/pci/devices/$bdf/current_link_speed"
cat "/sys/bus/pci/devices/$bdf/current_link_width"
cat "/sys/bus/pci/devices/$bdf/max_link_speed"
cat "/sys/bus/pci/devices/$bdf/max_link_width"
```

If current width/speed is lower than maximum, check the motherboard slot,
BIOS lane allocation, riser/cable, and competing devices. A failure may also
be resolved by changing rank order. Lowering the configured floor means
accepting an unqualified slower topology; it does not make the link faster.

## Host RAM

### `R9V_MIN_HOST_RAM_BYTES`

Optional hard floor for installed RAM. `0` means report without rejecting.

### `R9V_MIN_HOST_AVAILABLE_BYTES`

Optional hard floor for memory available immediately before launch. This is
useful on hosts that run other large services. `0` reports without rejecting.

Inspect the values with:

```bash
grep -E '^(MemTotal|MemAvailable):' /proc/meminfo
```

`/proc/meminfo` reports KiB, while R9V settings use bytes. Examples:

```bash
numfmt --from=iec 96Gi   # 103079215104
numfmt --from=iec 32Gi   # 34359738368
```

Example policy for requiring at least 96 GiB installed and 32 GiB available:

```bash
: "${R9V_MIN_HOST_RAM_BYTES:=103079215104}"
: "${R9V_MIN_HOST_AVAILABLE_BYTES:=34359738368}"
```

The profile was qualified on approximately 128 GB of host RAM. Smaller hosts
are not automatically rejected because the correct floor depends on other
processes and startup behavior. Cold expert allocations are pinned and do not
silently spill to SSD. Less RAM primarily reduces startup headroom and the
filesystem page cache available to the PLE file.

### CPU-offload values are accounting budgets

`R9V_CPU_OFFLOAD_GB=112.5` and
`R9V_CPU_OFFLOAD_GB_BY_DEVICE=112.5,112.5` are logical BF16 loader-accounting
budgets used to admit the expert tensors. They do not allocate 112.5 GiB per
rank. Do not lower them merely because the host has 96 GiB of RAM; doing so
can prevent the loader from offloading all experts.

## PLE/n-gram storage

### `R9V_PLE_PATH`

Absolute host path to the derived 28,800,138,240-byte IQ4_NL PLE table. It is
mounted read-only into the server.

```bash
: "${R9V_PLE_PATH:=/fast-ssd/r9v/per_layer_token_embd.iq4_nl.bin}"
stat -c '%n %s bytes' "$R9V_PLE_PATH"
```

If the size check fails, regenerate this derived file from the verified target
shards. Do not pad, truncate, or repair it manually.

### `R9V_PLE_EXPECTED_SHA256`

The size check catches a truncated extraction but not a corrupt one. This
optional setting records the sha256 of the derived PLE table so a bad
extraction fails at doctor time instead of surfacing later as unexplained
decode latency. The published default is in [`profile.env`](profile.env):

```bash
: "${R9V_PLE_EXPECTED_SHA256:=dd55c28902f38cd88134b2a569c51282c5ffce30080487e1a645740115c56cc3}"
```

Hashing reads the whole 26.82 GiB file and takes minutes, so it is not part of
a normal doctor run. Request it explicitly:

```bash
./r9v doctor qwen38 --model-dir "$R9V_MODEL_DIR" --hash-ple
```

| Configuration | Without `--hash-ple` | With `--hash-ple` |
|---|---|---|
| Empty or unset | `NOTE`: verified by size only | `NOTE`: verified by size only |
| 64 hex characters | `NOTE`: configured but unverified this run | `PASS` on match, `FAIL` on mismatch |
| Any other value | `FAIL`: not 64 hexadecimal characters | `FAIL`: not 64 hexadecimal characters |

Verify the file manually with:

```bash
sha256sum "$R9V_PLE_PATH"
```

A mismatch means the derived payload is wrong, not that the expectation is
wrong. Delete only this derived file and regenerate it from the verified target
shards. Never hand-repair the payload, and do not update the expected hash to
match a file whose provenance you have not re-established.

### `R9V_PLE_RESIDENCY_MODE`

| Value | Behavior | Release status |
|---|---|---|
| `ssd` | File-backed table with explicit trim/readahead policy. | Published default and benchmarked path |
| `bounded` | Keeps a bounded set of file pages resident and evicts cold ranges. | Advanced/experimental |
| `pinned` | Registers the full 26.82 GiB mapping as pinned host memory. | Diagnostic/advanced; requires substantial RAM |

Use `ssd` unless deliberately qualifying another policy:

```bash
: "${R9V_PLE_RESIDENCY_MODE:=ssd}"
```

This setting affects the PLE/n-gram table, not the prompt KV cache and not the
cold expert allocations.

### `R9V_REQUIRE_PLE_NONROTATIONAL`

`1` rejects a PLE backed by a rotating disk. The doctor follows LVM/device
mapper layers to the physical disk and reports its transport.

```bash
: "${R9V_REQUIRE_PLE_NONROTATIONAL:=1}"
findmnt -T "$R9V_PLE_PATH"
source=$(findmnt -n -o SOURCE -T "$R9V_PLE_PATH")
lsblk -s -o NAME,PATH,TYPE,ROTA,TRAN "$source"
```

`ROTA=0` means non-rotating. `TRAN=nvme` is the qualified storage class. A
SATA/USB SSD may pass the non-rotating requirement but produces a warning
because its random-read latency is unqualified. Move the PLE to NVMe or test
that device explicitly. Do not run the cold-page PLE benchmark while a live
server uses the same file; its cache-eviction calls intentionally disturb the
system page cache.

### `R9V_PLE_WORKER_TIMING`

Set this to `1` for one diagnostic relaunch when prefill is fast but decode is
slow:

```bash
: "${R9V_PLE_WORKER_TIMING:=1}"
```

After sending a generation request, runtime doctor reports the latest split
for input wait, n-gram ID construction, row gather/dequantization, output copy,
forward work, H2D enqueue, and total time. It only adds timing logs; it does
not move the table into RAM. Return it to `0` after collecting evidence.

## Expert placement and dynamic cache

The published manifest contains 329 static experts per layer on rank 0 and
369 on rank 1. Rank 1 also receives a 16-slot dynamic LRU cache, giving
effective maxima of 329 and 385.

### `R9V_TIERED_EXPERT_CACHE_RANKS`

Comma-separated TP ranks receiving a cache. Published value: `1`.

### `R9V_TIERED_EXPERT_CACHE_SLOTS`

Number of dynamic slots per selected rank. Supported range: 0 through 16.
Published value: `16`.

### `R9V_TIERED_EXPERT_CACHE_POLICY`

Published value: `lru`. Other policies are not part of the release arm.

### `R9V_MAX_EFFECTIVE_EXPERTS_PER_RANK`

Reference count ceiling for `maximum per-layer static count + physical cache
slots` on each rank. This is not a complete VRAM budget:

```bash
: "${R9V_MAX_EFFECTIVE_EXPERTS_PER_RANK:=329,385}"
```

The doctor validates the actual per-layer lists and rejects count-ceiling
violations, then reports expert byte costs separately. Passing this check does
not prove total VRAM fit. Do not raise the ceiling to silence a failure. Reduce cache slots or use a compatible
manifest whose static count leaves enough VRAM. Changing static expert counts
requires generating and qualifying a different manifest; it cannot be safely
expressed by changing only an environment variable.

The slowest unique PCIe rank normally benefits most from the dynamic cache.
If the cache is assigned elsewhere, doctor warns. That warning may be accepted
only after a controlled benchmark with the same manifest and source build.

## API address

| Setting | Published value | Meaning |
|---|---|---|
| `R9V_HOST_PORT` | `8004` | Host port of the OpenAI-compatible API: `http://127.0.0.1:8004/v1`. |
| `R9V_HOST_BIND` | `127.0.0.1` | Host address the API is published on: any IPv4 or IPv6 address. `0.0.0.0` means all IPv4 interfaces. |

The API has no authentication, so by default only programs on this machine can
reach it. To expose it, set `R9V_HOST_BIND` when you run setup; setup saves it
with the port, and running setup again changes either. With `R9V_CONFIG_FILE`,
set it in that file instead.

```bash
R9V_HOST_BIND=0.0.0.0 ./r9v setup qwen38-mtp4 --model-dir "$MODEL_DIR" \
  --accept-model-license
```

The launcher refuses a value that is not an IP address, and a port that
contains an address, before anything starts, and it warns whenever the API is
published beyond this machine. Start's health check, doctor and qualification
connect to `127.0.0.1`, so a specific address is published in addition to
`127.0.0.1`. Put your own authentication or firewall in front of an exposed
API. Exposing `qwen38-mtp4-uncensored` gives anyone who can reach the port a
model with no refusals; add your own authentication and moderation first.

## Context and VRAM settings

These are configurable but coupled. The published values are one qualified
128K, one-sequence configuration:

| Setting | Published value | Meaning |
|---|---:|---|
| `R9V_MAX_MODEL_LEN` | `131072` | Maximum prompt plus output context |
| `R9V_KV_CACHE_MEMORY_BYTES` | `2285670400` | Per-rank BF16 QSA/state cache reservation |
| `R9V_MAX_NUM_SEQS` | `1` | Maximum simultaneous sequences |
| `R9V_MAX_NUM_BATCHED_TOKENS` | `1024` | Scheduler/prefill token budget |

Increasing context, cache bytes, concurrency, or hot experts can change VRAM
headroom and graph capture. Decreasing KV bytes without reducing maximum
context can make startup reject the profile. Treat these as a coupled profile,
not independent performance sliders.

## MTP and kernel settings

The following values identify the published performance/correctness arm. They
are exposed for diagnostics and development, but changing them forfeits the
published benchmark comparison:

| Setting | Published value |
|---|---|
| `R9V_MTP_SPEC_TOKENS` | `2` |
| `R9V_MTP_DRAFT_TP_SIZE` | `2` |
| `R9V_MTP_LOCAL_ARGMAX` | `1` |
| `R9V_TIERED_IQ_MOE_VARIANT` | `reuse3v2` |
| `R9V_TIERED_PREFILL_GROUP_SIZE` | `16` |
| `R9V_DENSE_Q8_ATTN_M3_VARIANT` | `exact4-w8` |
| `R9V_ENABLE_DENSE_Q8_ATTN_M3` | `1` |
| `R9V_ENABLE_DENSE_HC_DOWN_BF16_M3` | `1` |
| `R9V_ENABLE_FUSED_HC_UP_MIX` | `1` |
| `R9V_ENABLE_FUSED_MOE_SHARED_EPILOGUE` | `1` |
| `R9V_ENABLE_FUSED_GDN_MTP` | `1` |
| `R9V_ENABLE_RDNA4_QSA_STRIDED` | `1` |

Every R4D setting must remain `0`; the launcher rejects any attempt to enable
it. R4D is not required for the published performance and is deliberately
excluded from this release.

## Static and runtime checks

Static doctor verifies:

- Docker, ROCm device nodes, source/submodule inputs, and two `gfx1201` GPUs.
- HIP index to BDF to TP-rank mapping.
- Exact PCIe path-capacity bottlenecks when configured, plus the slowest-hop payload
  of each rank's full path to the root port against per-rank floors.
- Total/available host RAM policy.
- Cache rank/slots and static-manifest-plus-cache VRAM ceiling.
- PLE payload size, filesystem, and physical media, plus its sha256 when
  `--hash-ple` is requested.
- Model-package artifact sizes.

Runtime doctor additionally verifies:

- The selected container is running from a concrete image ID.
- The JSON report retains its exit code, Docker `OOMKilled` flag, start/finish
  timestamps, restart count, and runtime error, including for stopped containers.
- Critical container environment values match the current profile/config.
- Tiered experts materialized on both TP ranks.
- The `reuse3v2` decode variant was selected.
- Grouped-16 prefill has been observed after a prompt longer than 64 tokens.
- Fused speculative GDN enable evidence.
- PLE timing evidence when requested.
- Speculative draft, accepted-token, acceptance-rate, and mean emitted-length
  counters.

The runtime report distinguishes two commonly confused cases:

- Low MTP acceptance reduces emitted tokens per verification cycle.
- Healthy MTP with slow TG points to cycle latency, commonly PCIe/expert
  traffic or PLE random-I/O latency.

## Common corrections

### Requests hang or the container fails after prolonged serving

Run this on the affected host while using the server normally. It observes the
existing container; it does not generate requests or restart anything:

```bash
r9v_support_dir="${XDG_STATE_HOME:-$HOME/.local/state}/r9v/support"
mkdir -p "$r9v_support_dir"
python3 tools/capture_runtime.py \
  --container "${R9V_CONTAINER_NAME:-r9v-qwen38-flash-next}" \
  --port "${R9V_HOST_PORT:-8004}" --duration 7200 --interval 30 \
  --output "$r9v_support_dir/runtime-$(date -u +%Y%m%dT%H%M%SZ).jsonl"
```

The default two-hour capture is capped at 32 MiB and refuses to overwrite
existing evidence. Keep the terminal open until it finishes, or interrupt
with Ctrl-C to retain the samples collected so far. Each sample is flushed
to disk and contains UTC time, container lifecycle/restart state, Docker
memory/CPU/process counts, cgroup v2 memory and OOM counters when readable,
host memory/swap pressure, selected aggregate request/token metrics, and
current/maximum PCIe speed and width plus AER counters for each AMD GPU path.
All GPU paths are identified by BDF; compare them with doctor's selected ranks.
Docker and HTTP probes have time/output limits and record unavailable data.

`possible_request_stall` means pending work showed no change in available
token/completion counters for three minutes; a long prefill can also trigger
it. An idle server is not classified as stalled. A Docker OOM flag is recorded
separately from exit code 137. If the whole host freezes, the last flushed
samples remain useful, but a truncated timeline alone cannot prove a host
crash. The collector includes bounded timestamped server/kernel log windows and GPU
VRAM, temperature, fan, and power readings where sysfs exposes them. It flags
increases in host/cgroup OOM and uncorrectable PCIe counters against the first
sample. It does not record API response bodies or full environments, but raw
logs may contain private data: review them before sharing. Busy log windows
can be truncated; truncation and inaccessible probes are explicit.

### Container exited or a user reports a crash

Collect evidence before removing or recreating the container:

```bash
./r9v support qwen38 --output /path/to/evidence/crash-20260907
```

Use a new directory each time. The bundle includes bounded server logs,
current and previous boot kernel logs (when journal permissions and retention
allow), container image/lifecycle, GPU/PCIe state, host/cgroup memory, metrics,
and runtime versions. Probe errors are saved even if Docker or the server is
unavailable. Collection is local and never uploads anything. Files are private
by default; review raw logs before sharing. Each command has a 10-second,
256-KiB bound. A truncated command keeps partial text and an explicit error.

The launcher retains the stopped container and explicitly rotates Docker JSON
logs across five 20-MiB files. It enables Python unbuffered output and fatal
fault tracebacks. Removing the container deletes its Docker logs; collect the
bundle first. A host reboot requires persistent journald storage to recover
previous-boot kernel events. This is checked, not configured, by doctor.

Include the failure's UTC timestamp, whether anyone stopped the container,
the request shape/token counts, and host kernel logs covering that time.
An exit code of 137 alone does not establish an OOM kill. A SIGTERM followed
by forced worker cleanup records a shutdown sequence; it does not identify
who initiated the stop or prove an inference crash. Correlate the container
state with engine and kernel evidence before changing memory or kernel settings.

### Repeatable soak testing

Start with a ready server and run:

```bash
./r9v doctor qwen38 --runtime --json > /path/to/evidence/doctor-before.json
./r9v soak qwen38 --duration 7200 --request-timeout 300 \
  --output /path/to/evidence/soak-two-hours
```

The default sequential workload cycles short, medium, and long synthetic text
prompts, then idles for five seconds. Every request changes its prefix to avoid
reusing only cached prompts. `--prompt-repeats 8,128,2048` changes corpus sizes;
these are repeat counts, not exact token lengths. Token usage is recorded from
the server. `--idle-seconds 600` exercises longer idle/wake cycles. Use
`--duration 28800` for eight hours and `86400` for 24 hours. The last in-flight
request can extend the duration by at most `--request-timeout`, followed by
bounded support collection. For long prefills, increase the timeout explicitly.

`requests.jsonl` saves a durable record before and after each request;
`timeline.jsonl` independently samples telemetry every ten seconds;
`summary.json` records pass/fail/interruption; `support/` captures final crash
evidence. A request timeout, HTTP failure, malformed/empty completion,
container restart/exit, unavailable progress metrics, possible stall, new OOM,
or new uncorrectable PCIe counter prevents a pass. Historical counters are
retained without being called a new failure. At least one complete corpus-size
cycle is required. Ctrl-C preserves evidence and returns failure/interrupted.

A successful soak establishes liveness for this workload only. It does not
verify semantic correctness, streaming/tool/vision paths, or exact 128K context.
Run those qualification cases separately. A power loss may prevent a summary;
the fsynced request and timeline records still identify the unfinished trial.
Keep the terminal open, or use your service manager to run the command across
terminal disconnects. The harness never restarts or removes the server.

### Resource checks before declaring a host ready

Doctor now fails when model or PLE paths are unset. It checks physical-card
VRAM, warns about occupied VRAM before launch, cache storage, recorded PCIe
errors, and absent measured RAM gates. Its byte accounting reports packed hot/cold/cache storage and the pageable
loading-master component separately. It does not infer a host peak from target
file sizes: the shards include file-backed PLE, and loading copies, MTP and
page-cache behavior must be measured. Runtime
checks also identify container memory/PID limits, memlock, automatic removal,
and log retention. Missing startup markers in recent/rotated logs are warnings
about missing evidence, not proof of a bad kernel selection.

Use `--strict` for qualification once warnings are resolved or accounted for.
A default doctor exit of zero means no hard failure; review WARN checks as well.
Two cards and installed RAM alone do not prove sufficient free startup memory
or a healthy PCIe path. Publish measured RAM minima only after clean startup
and sustained trials on the smallest supported host.

See [the ROCm 10.0 qualification plan](../../../docs/qualification/qwen38-rocm10-stability.md)
for the upgrade and release gates.

### Similar machine, defaults work

Run static doctor once without BDF or PCIe-link locks. Confirm its detected
order and topology, paste the suggested `R9V_EXPECTED_GPU_BDFS` and
`R9V_EXPECTED_PCIE_LINKS` lines into the config, then rerun until the only
remaining warnings are understood.

### GPUs are enumerated in the wrong order

Set `R9V_VISIBLE_DEVICES` in the intended HIP/KFD rank order, leave the BDF
lock empty, and run the doctor. Confirm its resolved HIP-index-to-BDF mapping,
then copy the suggested `R9V_EXPECTED_GPU_BDFS` value and rerun before
launching. Never translate a numeric HIP index by looking up the same numeric
row in `amd-smi list`.

### PCIe bandwidth is below the configured floor

Compare current and maximum link speed/width in sysfs. Check slot wiring, BIOS
lane bifurcation, risers, and rank order. Lower the floor only if the topology
is intentional and you accept that the published TG figure may not apply.

### PCIe link does not match the exact lock

The exact lock reports stable path capacity; it does not try to tune hardware.
Compare `current_link_*` and `max_link_*` in sysfs. If current is below max,
check slot wiring, BIOS bifurcation, risers/cables, and devices sharing lanes.
If the detected link is the topology you intentionally built, update
`R9V_EXPECTED_PCIE_LINKS`; keep the minimum-bandwidth floor high enough for the
performance level you intend to qualify.

### Host has 96 GiB RAM

Keep PLE mode on `ssd`, leave CPU-offload accounting at 112.5, close other
memory-heavy services, and use doctor to inspect available RAM. Enable PLE
timing for the first performance qualification because reduced page cache can
increase decode latency.

### Fast PP but unexpectedly slow TG

Prefill and decode use different paths. Do not infer decode health from PP.
Enable PLE worker timing, relaunch, send a normal generation request, and run:

```bash
./r9v doctor qwen38 --model-dir "$R9V_MODEL_DIR" --runtime --json \
  > r9v-qwen-runtime.json
```

The report proves whether the tiered decode variant loaded, whether both ranks
materialized the placement, whether MTP is drafting/accepting tokens, and
whether PLE work is consuming the cycle.

### Expert budget exceeds the maximum

Do not raise `R9V_MAX_EFFECTIVE_EXPERTS_PER_RANK`. Reduce cache slots or select
a manifest built for the requested cache capacity. Static expert placement,
cache slots, KV reservation, and display headroom all consume the same VRAM.

## Bypasses

`R9V_PREFLIGHT=0` bypasses automatic launch preflight and prints a warning.
This exists for development recovery, not normal use. It does not make an
incompatible configuration safe and forfeits support/qualification evidence.

## Per-rank headroom and experimental smaller placements

Set a desired physical free-memory margin in GiB, in TP-rank order:

```bash
export R9V_MIN_FREE_VRAM_GIB_BY_RANK=5,5
```

Doctor checks both cards and rejects an already-impossible expert + KV + margin
budget. This setting is a check, **not automatic expert resizing** or a promise
about future peak allocations. Default margins are 3 GiB on both ranks. Do not
interpret passing count ceilings or an idle snapshot as total-memory admission.

To derive a smaller manifest without editing the verified model package:

```bash
./r9v placement qwen38 --model-dir "$R9V_MODEL_DIR" \
  --hot-counts 329,329 --output /path/to/new-experts.json
export R9V_EXPERT_MANIFEST_PATH=/path/to/new-experts.json
./r9v doctor qwen38 --model-dir "$R9V_MODEL_DIR"
```

The launcher mounts that external file read-only. With the existing rank-1
LRU16 configuration, the example frees 2.165 GiB of packed GPU weights on rank 1
and adds 2.165 GiB to pinned host RAM; rank 0 is unchanged. Cache settings remain
separate. The tool only trims supplied priority prefixes, never overwrites a
file, removes stale route statistics, and marks the result unqualified. It
cannot grow the truncated source map or guarantee five free GiB on either card.
Qualify correctness, memory peaks and throughput before choosing a new default.

`R9V_MIN_PCIE_BANDWIDTH_GBPS` now defaults to `0,0`: a complete slower topology
is not automatically an invalid setup. `R9V_REFERENCE_PCIE_BANDWIDTH_GBPS=15,7`
still warns below reference performance. Set explicit hard floors if you want
to reject slower links; exact BDF/link locks and invalid path detection remain.

See [the hardware and headroom review](../../../docs/qualification/qwen38-headroom-design.md)
for the full ranked-catalog/planner design, non-memory crash candidates and the
remaining worker-level checks.

## Uncensored profile: runtime overlays, CED and fixed placement

`qwen38-mtp4-uncensored` (`profiles/qwen38-flash-next/dual-r9700-mtp4-uncensored/profile.env`)
uses these settings on top of the ones above. Setup saves them; change CED with
`--ced on|off` rather than by editing the saved configuration.

| Setting | Profile value | Meaning |
|---|---|---|
| `R9V_RUNTIME_DESCRIPTOR` | `runtimes/qwen38-flash-next-gfx1201-mtp4-v3/runtime.json` | Runtime whose `overlays` block lists the files mounted read-only over the image, with their SHA-256. The launcher refuses to start if any file differs. |
| `R9V_CED` | `on` | CED (approximate long-prompt prefill). `off` mounts neither the CED model file nor any `R9V_CED_*` setting. |
| `R9V_CED_PROJECTOR_REL` | `ced/ced-projector-split16.safetensors` | Projector inside the model directory, read at `/models/…`. |
| `R9V_CED_PRECISION` | `bf16` | `int8` roughly halves the projector's 1.76 GiB per GPU but was never run on the GPU. |
| `R9V_CED_MIN_PROMPT` | `8192` | With `R9V_CED_DEFAULT=on`, shorter prompts stay exact. At least 0. |
| `R9V_CED_TAIL` | `2048` | Exact prompt tail in tokens, rounded up to a cache block. At least 512; only 2048 is graded. |
| `R9V_CED_DEFAULT` | `on` | `on`: requests that do not say get CED above the minimum. `off`: only requests with `r9v_ced: true`. |
| `R9V_PREFIX_CACHE_RETENTION_INTERVAL` | `1616` | Prefix-cache checkpoint interval, the scheduler's block boundary on this image. |
| `R9V_MIN_FREE_VRAM_GIB_BY_RANK` | `1.5,1.5` | Lowered from 3,3 because the loaded projector takes 1.76 GiB per GPU. Provisional until the clean-host qualification measures the workload minimum. |
| `R9V_MIN_HOST_AVAILABLE_BYTES` | `76699664384` | Available RAM before launch: every expert pinned on the host (59.5 GB) plus the 16 GiB PLE reserve. |
| `R9V_EXPERT_MANIFEST_PATH` | the fixed `mtp4-warmstart-r1/manifest.json` | Pinned by SHA-256 in `mtp4-full-mutable.json`. The mutable expert cache refuses any other placement. |

Requests choose per call with `vllm_xargs`: `r9v_ced` (`true` forces CED even
below the minimum, `false` keeps the request exact) and `r9v_ced_tail` (this
request's exact tail). Prompts with images and requests for prompt logprobs run
exactly. CED requests keep their own prefix cache, so an exact request never
reuses a CED request's cache.

Because the placement is fixed, `--headroom`, `--calibration` and
`--expert-catalog` are refused, and the `plan` and `placement` commands are not
available for this profile. First start qualifies the placement; the receipt
is keyed on the manifest, runtime descriptor, image and CED setting.
