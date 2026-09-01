# Qwen3.8 Flash Next dual-R9700 configuration

This is the machine-configuration reference for the published Qwen3.8 Flash
Next R9V profile. The general installation sequence is in
[`docs/installation.md`](../../../docs/installation.md); this page explains
what each host-facing setting means, how to discover the correct value, and
how to respond to every class of doctor result.

The published profile is optimized for two 32 GiB Radeon AI PRO R9700 GPUs.
The GPUs do not need to occupy the same PCIe slots as the reference machine.
The rank order, negotiated link bandwidth, RAM policy, PLE storage, and expert
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

Discover indices and PCI addresses with:

```bash
amd-smi list
```

Example:

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

This optional lock records the exact negotiated generation/speed and lane
width expected for each TP rank. It accepts a readable generation form or a
numeric transfer-rate form:

```bash
: "${R9V_EXPECTED_PCIE_LINKS:=Gen5x16,Gen4x4}"
# Equivalent:
: "${R9V_EXPECTED_PCIE_LINKS:=32x16,16x4}"
```

The values follow `R9V_VISIBLE_DEVICES` rank order. Leave the setting empty on
the first doctor run. If sysfs exposes both links, doctor prints the exact line
to paste into the configuration. Once locked, a speed or width change fails
preflight instead of silently changing the performance profile.

R9V cannot set a physical PCIe generation or lane width. Those are negotiated
by the GPU, motherboard slot, BIOS lane allocation, riser/cable, and competing
devices. This setting only verifies the result. If it fails, repair the
hardware/firmware topology or deliberately update the lock after deciding the
new topology is correct.

The lock matches the endpoint's own negotiated link. Upstream-hop capacity is
enforced separately by the bandwidth floor below, which walks the full path.

### `R9V_MIN_PCIE_BANDWIDTH_GBPS`

This is the minimum theoretical one-direction PCIe payload bandwidth for each
TP rank, not a benchmark result. The doctor reads the negotiated speed and
width of every hop from the device up to the root port, accounts for PCIe
link encoding, and applies the floor to the slowest hop. An endpoint that
negotiates x16 behind a x4 upstream bridge is therefore scored at the x4
bottleneck, and the report names the capping hop. Endpoint sysfs alone cannot
be trusted for this: a card can read Gen5x16 at the endpoint while an
upstream switch or bifurcated link caps the real path.

The published defaults are:

```bash
: "${R9V_MIN_PCIE_BANDWIDTH_GBPS:=15,7}"
```

They allow topology shapes different from the reference host when their
payload capability is equivalent or better. This bandwidth floor is
independent of the exact link lock: use the floor to express "fast enough"
and the lock to express "the intended slots negotiated as expected." Inspect
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

VRAM safety ceiling for `manifest static count + configured cache slots` on
each rank:

```bash
: "${R9V_MAX_EFFECTIVE_EXPERTS_PER_RANK:=329,385}"
```

The doctor reads the actual selected manifest and rejects overcommit. Do not
raise this value to silence a failure. Reduce cache slots or use a compatible
manifest whose static count leaves enough VRAM. Changing static expert counts
requires generating and qualifying a different manifest; it cannot be safely
expressed by changing only an environment variable.

The slowest unique PCIe rank normally benefits most from the dynamic cache.
If the cache is assigned elsewhere, doctor warns. That warning may be accepted
only after a controlled benchmark with the same manifest and source build.

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
- Exact negotiated PCIe links when configured, plus the slowest-hop payload
  of each rank's full path to the root port against per-rank floors.
- Total/available host RAM policy.
- Cache rank/slots and static-manifest-plus-cache VRAM ceiling.
- PLE payload size, filesystem, and physical media, plus its sha256 when
  `--hash-ple` is requested.
- Model-package artifact sizes.

Runtime doctor additionally verifies:

- The selected container is running from a concrete image ID.
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

### Similar machine, defaults work

Run static doctor once without BDF or PCIe-link locks. Confirm its detected
order and topology, paste the suggested `R9V_EXPECTED_GPU_BDFS` and
`R9V_EXPECTED_PCIE_LINKS` lines into the config, then rerun until the only
remaining warnings are understood.

### GPUs are enumerated in the wrong order

Use `amd-smi list`, set `R9V_VISIBLE_DEVICES` in the intended rank order, and
set `R9V_EXPECTED_GPU_BDFS` in that same order. Rerun static doctor before
launching.

### PCIe bandwidth is below the configured floor

Compare current and maximum link speed/width in sysfs. Check slot wiring, BIOS
lane bifurcation, risers, and rank order. Lower the floor only if the topology
is intentional and you accept that the published TG figure may not apply.

### PCIe link does not match the exact lock

The exact lock is reporting what the system negotiated, not trying to tune it.
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
