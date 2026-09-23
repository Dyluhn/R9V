#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
profile=${R9V_PROFILE:-$repo_root/profiles/qwen38-flash-next/dual-r9700/profile.env}
[[ -r "$profile" ]] || { printf 'Profile not found: %s\n' "$profile" >&2; exit 1; }
config_file=${R9V_CONFIG_FILE:-}
if [[ -n $config_file ]]; then
    [[ -r $config_file ]] || {
        printf 'R9V_CONFIG_FILE is not readable: %s\n' "$config_file" >&2
        exit 1
    }
    set -a
    # shellcheck disable=SC1090
    source "$config_file"
    set +a
fi
set -a
# shellcheck disable=SC1090
source "$profile"
set +a

image=${R9V_IMAGE:-r9v-qwen38-flash-next:latest}
container=${R9V_CONTAINER_NAME:-r9v-qwen38-flash-next}
model_dir=${R9V_MODEL_DIR:?Set R9V_MODEL_DIR to the packaged model directory}
ple_path=${R9V_PLE_PATH:?Set R9V_PLE_PATH to the extracted PLE payload}
cache_dir=${R9V_CACHE_DIR:-$repo_root/.cache}
visible_devices=${R9V_VISIBLE_DEVICES:-0,1}
[[ $visible_devices =~ ^[0-9]+(,[0-9]+)*$ ]] || {
    printf 'R9V_VISIBLE_DEVICES requires numeric HIP/KFD indices; AMD-SMI UUIDs are not ROCR UUIDs. Confirm cards with R9V_EXPECTED_GPU_BDFS.\n' >&2
    exit 2
}
# ROCR selects host GPU ordinals; HIP sees the resulting local ordinal space.
# Applying the same nontrivial selection twice can reverse ranks or hide a GPU.
IFS=',' read -r -a selected_devices <<< "$visible_devices"
printf -v hip_visible_devices '%s,' "${!selected_devices[@]}"
hip_visible_devices=${hip_visible_devices%,}
: "${R9V_PREFLIGHT:=1}"
: "${R9V_TIERED_PREFILL_GROUP_SIZE:=0}"
: "${R9V_PLE_PINNED_RESERVE_BYTES:=17179869184}"
: "${R9V_PLE_HOST_FENCE:=1}"
: "${R9V_DEV_FUSED_MOE_PY:=}"
: "${R9V_DEV_LINEAR_PY:=}"
: "${R9V_DEV_TIERED_IQ_MOE_SO:=}"

[[ $R9V_TIERED_PREFILL_GROUP_SIZE == 0 ||
   $R9V_TIERED_PREFILL_GROUP_SIZE == 4 ||
   $R9V_TIERED_PREFILL_GROUP_SIZE == 8 ||
   $R9V_TIERED_PREFILL_GROUP_SIZE == 16 ||
   $R9V_TIERED_PREFILL_GROUP_SIZE == 32 ]] || {
    printf 'R9V_TIERED_PREFILL_GROUP_SIZE must be 0, 4, 8, 16, or 32\n' >&2
    exit 2
}

case $R9V_PLE_RESIDENCY_MODE in
    ssd|bounded)
        ple_mmap_host_register=0
        ;;
    pinned)
        ple_mmap_host_register=1
        ;;
    *)
        printf 'R9V_PLE_RESIDENCY_MODE must be ssd, pinned, or bounded\n' >&2
        exit 2
        ;;
esac
[[ $R9V_PLE_PINNED_RESERVE_BYTES =~ ^[0-9]+$ ]] || {
    printf 'R9V_PLE_PINNED_RESERVE_BYTES must be a non-negative integer\n' >&2
    exit 2
}
[[ $R9V_PREFLIGHT == 0 || $R9V_PREFLIGHT == 1 ]] || {
    printf 'R9V_PREFLIGHT must be 0 or 1\n' >&2
    exit 2
}

dev_overlay_args=()
# Diagnostic serialization localizes asynchronous HIP failures. Off by default.
if [[ ${R9V_HIP_FAULT_DIAGNOSTICS:-0} == 1 ]]; then
    # HIP parses integer flags with atoi: use decimal, not a 0x-prefixed mask.
    # API calls, commands, and shader names; the supervisor must bound log storage.
    dev_overlay_args+=(--env AMD_SERIALIZE_KERNEL=3 --env AMD_SERIALIZE_COPY=3
                      --env AMD_LOG_LEVEL=4 --env AMD_LOG_MASK=2097155)
elif [[ ${R9V_HIP_FAULT_DIAGNOSTICS:-0} != 0 ]]; then
    printf 'R9V_HIP_FAULT_DIAGNOSTICS must be 0 or 1\n' >&2
    exit 2
fi
# Opt-in source overlays for host-only observability on an immutable test image.
for entry in R9V_DEV_GPU_WORKER_PY:gpu_worker.py R9V_DEV_WORKER_DIAGNOSTICS_PY:r9v_diagnostics.py; do
    key=${entry%%:*}
    filename=${entry#*:}
    source_path=${!key:-}
    if [[ -n $source_path ]]; then
        [[ -f $source_path ]] || { printf 'Observability source missing: %s\n' "$source_path" >&2; exit 1; }
        dev_overlay_args+=(--volume "$source_path:/opt/r9v/lib/python3.12/site-packages/vllm/v1/worker/$filename:ro")
    fi
done
if [[ -n ${R9V_DEV_SHORT_CONV_PY:-} ]]; then
    [[ -f $R9V_DEV_SHORT_CONV_PY ]] || {
        printf 'Development short_conv_attn.py missing: %s\n' "$R9V_DEV_SHORT_CONV_PY" >&2
        exit 1
    }
    dev_overlay_args+=(
        --volume "$R9V_DEV_SHORT_CONV_PY:/opt/r9v/lib/python3.12/site-packages/vllm/v1/attention/backends/short_conv_attn.py:ro"
    )
fi
if [[ -n $R9V_DEV_FUSED_MOE_PY ]]; then
    [[ -f $R9V_DEV_FUSED_MOE_PY ]] || {
        printf 'Development fused_moe.py missing: %s\n' "$R9V_DEV_FUSED_MOE_PY" >&2
        exit 1
    }
    dev_overlay_args+=(
        --volume "$R9V_DEV_FUSED_MOE_PY:/opt/r9v/lib/python3.12/site-packages/vllm_gguf_plugin/quantization/fused_moe.py:ro"
    )
fi
if [[ -n $R9V_DEV_LINEAR_PY ]]; then
    [[ -f $R9V_DEV_LINEAR_PY ]] || {
        printf 'Development linear.py missing: %s\n' "$R9V_DEV_LINEAR_PY" >&2
        exit 1
    }
    dev_overlay_args+=(
        --volume "$R9V_DEV_LINEAR_PY:/opt/r9v/lib/python3.12/site-packages/vllm_gguf_plugin/quantization/linear.py:ro"
    )
fi
if [[ -n $R9V_DEV_TIERED_IQ_MOE_SO ]]; then
    [[ -f $R9V_DEV_TIERED_IQ_MOE_SO ]] || {
        printf 'Development tiered MoE SO missing: %s\n' "$R9V_DEV_TIERED_IQ_MOE_SO" >&2
        exit 1
    }
    dev_overlay_args+=(
        --volume "$R9V_DEV_TIERED_IQ_MOE_SO:/opt/r9v/kernels/qwen38_tiered_iq_moe_hip.so:ro"
    )
fi

for value in \
    R9V_MTP_LOCAL_ARGMAX \
    R9V_ENABLE_AUTO_TOOL_CHOICE \
    R9V_PLE_MMAP_READAHEAD \
    R9V_PLE_WORKER_TIMING \
    R9V_PLE_HOST_FENCE \
    R9V_R4D \
    R9V_R4D_AR \
    R9V_R4D_GDN \
    R9V_R4D_AR_QUANT; do
    [[ ${!value} == 0 || ${!value} == 1 ]] || {
        printf '%s must be 0 or 1\n' "$value" >&2
        exit 2
    }
done

profiler_mount_args=()
profiler_args=()
profiler_scopes=0
if [[ -n $R9V_PROFILER_DIR ]]; then
    for value in \
        R9V_PROFILER_WAIT_ITERATIONS \
        R9V_PROFILER_WARMUP_ITERATIONS \
        R9V_PROFILER_ACTIVE_ITERATIONS \
        R9V_PROFILER_MAX_ITERATIONS; do
        [[ ${!value} =~ ^[0-9]+$ ]] || {
            printf '%s must be a non-negative integer\n' "$value" >&2
            exit 2
        }
    done
    ((R9V_PROFILER_ACTIVE_ITERATIONS > 0)) || {
        printf 'R9V_PROFILER_ACTIVE_ITERATIONS must be positive\n' >&2
        exit 2
    }
    mkdir -p -- "$R9V_PROFILER_DIR"
    profiler_mount_args=(--volume "$R9V_PROFILER_DIR:/profiles")
    profiler_args=(
        --profiler-config
        "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"/profiles\",\"torch_profiler_with_stack\":true,\"torch_profiler_record_shapes\":true,\"torch_profiler_with_memory\":false,\"ignore_frontend\":true,\"wait_iterations\":$R9V_PROFILER_WAIT_ITERATIONS,\"warmup_iterations\":$R9V_PROFILER_WARMUP_ITERATIONS,\"active_iterations\":$R9V_PROFILER_ACTIVE_ITERATIONS,\"max_iterations\":$R9V_PROFILER_MAX_ITERATIONS}"
    )
    profiler_scopes=1
fi
custom_scopes=${R9V_CUSTOM_SCOPES_FOR_PROFILING:-$profiler_scopes}
dense_scopes=${R9V_PROFILE_DENSE_SHAPES:-$profiler_scopes}
[[ $custom_scopes == 0 || $custom_scopes == 1 ]] || exit 2
[[ $dense_scopes == 0 || $dense_scopes == 1 ]] || exit 2
prefix_cache_args=()
case "${R9V_ENABLE_PREFIX_CACHING:-1}" in
    0) prefix_cache_args=(--no-enable-prefix-caching) ;;
    1) prefix_cache_args=(--enable-prefix-caching) ;;
    *) printf 'R9V_ENABLE_PREFIX_CACHING must be 0 or 1\n' >&2; exit 2 ;;
esac
if [[ -n ${R9V_PREFIX_CACHE_RETENTION_INTERVAL:-} && ${R9V_ENABLE_PREFIX_CACHING:-1} == 1 ]]; then
    [[ $R9V_PREFIX_CACHE_RETENTION_INTERVAL =~ ^[0-9]+$ ]] || {
        printf 'R9V_PREFIX_CACHE_RETENTION_INTERVAL must be a non-negative integer\n' >&2
        exit 2
    }
    prefix_cache_args+=(--prefix-cache-retention-interval "$R9V_PREFIX_CACHE_RETENTION_INTERVAL")
fi
route_args=()
eager_args=()
if [[ -n ${R9V_ROUTE_PROFILE_DIR:-} ]]; then
    mkdir -p -- "$R9V_ROUTE_PROFILE_DIR"
    route_args=(--volume "$R9V_ROUTE_PROFILE_DIR:/routes" --env RADIANCE_ROUTE_PROFILE_DIR=/routes
                --env RADIANCE_ROUTE_PROFILE_HISTOGRAM=1 --env RADIANCE_ROUTE_PROFILE_RANKS=0)
    eager_args=(--enforce-eager)
fi
for value in R9V_R4D R9V_R4D_AR R9V_R4D_GDN R9V_R4D_AR_QUANT; do
    [[ ${!value} == 0 ]] || {
        printf '%s is unsupported: every R4D path is hard-disabled\n' "$value" >&2
        exit 2
    }
done
local_argmax=false
[[ $R9V_MTP_LOCAL_ARGMAX == 0 ]] || local_argmax=true
auto_tool_args=()
[[ $R9V_ENABLE_AUTO_TOOL_CHOICE == 0 ]] || auto_tool_args+=(--enable-auto-tool-choice)

target_rel=${R9V_TARGET_REL:-target/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf}
target_shard2_rel=${R9V_TARGET_SHARD2_REL:-target/Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf}
target_shard3_rel=${R9V_TARGET_SHARD3_REL:-target/Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00003.gguf}
target_shard4_rel=${R9V_TARGET_SHARD4_REL:-}
metadata_rel=${R9V_METADATA_REL:-metadata}
mtp_rel=${R9V_MTP_REL:-mtp}
mmproj_rel=${R9V_MMPROJ_REL:-vision/mmproj-Qwen3.8-Flash-Next-Q8_0.gguf}
manifest_rel=${R9V_MANIFEST_REL:-manifests/hot-manifest-q4-vision-128k-multiprompt-r1-lru16-neutral.json}

# Zero disables MTP for correctness isolation; positive depths retain the profile path.
[[ $R9V_MTP_SPEC_TOKENS =~ ^[0-9]+$ ]] || {
    printf 'R9V_MTP_SPEC_TOKENS must be a non-negative integer\n' >&2
    exit 2
}
compilation_config=$(python3 - "$R9V_MTP_SPEC_TOKENS" <<'PYGRAPH'
import json, sys
depth = int(sys.argv[1])
if not 0 <= depth <= 8:
    raise SystemExit('MTP depth must be 0..8; additional graph shapes require explicit qualification')
sizes = sorted({1, depth + 1})
print(json.dumps({'cudagraph_mode': 'FULL_DECODE_ONLY', 'cudagraph_capture_sizes': sizes, 'max_cudagraph_capture_size': max(sizes)}))
PYGRAPH
)
retained_args=()
if [[ ${R9V_RETAINED_MTP4:-0} == 1 ]]; then
    retained_args+=(--env "QWEN38_EXPERT_TP_SPLIT=${R9V_EXPERT_TP_SPLIT:?Missing expert channel partition}")
    for retained_key in R9V_RETAINED_MTP4 R9V_DRAFT_W2_RANK_ROWS R9V_CACHE_FILL_BATCH \
        R9V_CACHE80 R9V_CACHE192 R9V_Q6K_HEAD_ROWS5 R9V_DENSE_ROWS45 R9V_DENSE_ROWS2 \
        R9V_HC_ROWS45 R9V_HC_MMQ4 R9V_HC_MIX_MMQ4 R9V_HC_MIX_MMQ45 \
        R9V_HC_SAFE_PADDING R9V_ATTENTION_OUTPUT4 R9V_SHARED_SAFE_PADDING \
        R9V_NATIVE_CUSTOM_AR R9V_NATIVE_ALIGN_IPC R9V_Q8_MMVQ5_SHAPES; do
        [[ -v $retained_key ]] || { printf 'Missing retained runtime setting: %s\n' "$retained_key" >&2; exit 2; }
        retained_args+=(--env "$retained_key=${!retained_key}")
    done
    retained_args+=(--env VLLM_ROCM_USE_AITER_CUSTOM_AR=0 --env VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=NONE)
fi

speculative_args=()
if [[ $R9V_MTP_SPEC_TOKENS != 0 ]]; then
    speculative_args=(--speculative-config "{\"method\":\"mtp\",\"model\":\"/models/$mtp_rel\",\"num_speculative_tokens\":$R9V_MTP_SPEC_TOKENS,\"draft_tensor_parallel_size\":$R9V_MTP_DRAFT_TP_SIZE,\"quantization\":\"$R9V_MTP_QUANTIZATION\",\"use_local_argmax_reduction\":$local_argmax,\"draft_load_config\":{\"load_format\":\"auto\"}}")
fi

manifest_path=${R9V_EXPERT_MANIFEST_PATH:-$model_dir/$manifest_rel}
manifest_container_path="/models/$manifest_rel"
manifest_mount_args=()
if [[ -n ${R9V_EXPERT_MANIFEST_PATH:-} ]]; then
    [[ $manifest_path == /* ]] || { printf 'R9V_EXPERT_MANIFEST_PATH must be absolute\n' >&2; exit 2; }
    manifest_container_path=/placement/experts.json
    manifest_mount_args=(--volume "$manifest_path:$manifest_container_path:ro")
fi

target_files=("$model_dir/$target_rel" "$model_dir/$target_shard2_rel" "$model_dir/$target_shard3_rel")
[[ -z $target_shard4_rel ]] || target_files+=("$model_dir/$target_shard4_rel")
for required in \
    "${target_files[@]}" \
    "$model_dir/$metadata_rel/config.json" \
    "$model_dir/$mtp_rel/config.json" \
    "$model_dir/$mtp_rel/model.safetensors" \
    "$model_dir/$mmproj_rel" \
    "$manifest_path" \
    "$ple_path"; do
    [[ -f "$required" ]] || { printf 'Required file missing: %s\n' "$required" >&2; exit 1; }
done

if [[ $R9V_PREFLIGHT == 1 ]]; then
    "$repo_root/scripts/profile-doctor.sh"
else
    printf 'WARN launch preflight is disabled by R9V_PREFLIGHT=0\n' >&2
fi

if docker container inspect "$container" >/dev/null 2>&1; then
    printf 'Container already exists: %s\nStop/remove it explicitly before relaunch.\n' \
        "$container" >&2
    exit 1
fi

mkdir -p "$cache_dir"
cache_namespace=${R9V_CACHE_NAMESPACE:-}
if [[ -z $cache_namespace ]]; then
    resolved_image=$(docker image inspect "$image" --format '{{.Id}}')
    cache_namespace=$(python3 "$repo_root/tools/runtime_cache_key.py" "$resolved_image" "$manifest_path")
fi
[[ $cache_namespace =~ ^[a-zA-Z0-9_-]{1,64}$ ]] || {
    printf 'Invalid R9V_CACHE_NAMESPACE\n' >&2
    exit 2
}
runtime_cache_root=/cache/vllm/$cache_namespace
[[ $cache_namespace != legacy ]] || runtime_cache_root=/cache/vllm
printf 'Runtime compile cache: %s\n' "$runtime_cache_root"
device_group_args=()
# Numeric device ownership also works on distributions without render/video groups.
for device in /dev/kfd /dev/dri/renderD* /dev/dri/card*; do
    [[ -e $device ]] || continue
    device_group_args+=(--group-add "$(stat -c %g "$device")")
done

# Rootless daemons cannot raise RLIMIT_MEMLOCK above their inherited hard
# limit. An unlimited request fails in runc before the server even starts.
memlock_args=(--ulimit memlock=-1:-1)
if [[ $(docker info --format '{{json .SecurityOptions}}') == *name=rootless* ]]; then
    memlock_args=()
    printf 'Rootless Docker: inheriting daemon memlock limits; runtime doctor reports the effective policy.\n'
fi

docker run --detach \
    --name "$container" \
    --log-driver json-file \
    --log-opt max-size=20m \
    --log-opt max-file=5 \
    "${memlock_args[@]}" \
    --env PYTHONUNBUFFERED=1 \
    --env PYTHONFAULTHANDLER=1 \
    --device /dev/kfd \
    --device /dev/dri \
    "${device_group_args[@]}" \
    --ipc host \
    --security-opt seccomp=unconfined \
    --security-opt label=disable \
    --publish "$R9V_HOST_PORT:8000" \
    --volume "$model_dir:/models:ro" \
    --volume "$ple_path:/ple/per_layer_token_embd.iq4_nl.bin:ro" \
    --volume "$cache_dir:/cache" \
    "${manifest_mount_args[@]}" \
    "${dev_overlay_args[@]}" \
    "${profiler_mount_args[@]}" \
    "${route_args[@]}" \
    "${retained_args[@]}" \
    --env HIP_VISIBLE_DEVICES="$hip_visible_devices" \
    --env ROCR_VISIBLE_DEVICES="$visible_devices" \
    --env VLLM_CACHE_ROOT="$runtime_cache_root" \
    --env TRITON_CACHE_DIR="$runtime_cache_root/bootstrap/triton" \
    --env TRITON_CACHE_AUTOTUNING=1 \
    --env TORCHINDUCTOR_CACHE_DIR="$runtime_cache_root/bootstrap/inductor" \
    --env TORCH_EXTENSIONS_DIR="$runtime_cache_root/extensions" \
    --env RADIANCE_CPU_OFFLOAD_GB_BY_DEVICE="$R9V_CPU_OFFLOAD_GB_BY_DEVICE" \
    --env R9V_CPU_OFFLOAD_GB_BY_DEVICE="$R9V_CPU_OFFLOAD_GB_BY_DEVICE" \
    --env RADIANCE_TIERED_EXPERT_MANIFEST="$manifest_container_path" \
    --env RADIANCE_UVA_HOST_COHERENCE=default \
    --env RADIANCE_UVA_HOST_NONCOHERENT=0 \
    --env RADIANCE_USE_R4D=0 \
    --env RADIANCE_USE_R4D_AR=0 \
    --env RADIANCE_USE_R4D_GDN=0 \
    --env RADIANCE_USE_R4D_AR_QUANT=0 \
    --env QWEN38_USE_TIERED_IQ_MOE_HIP=1 \
    --env QWEN38_TIERED_IQ_MOE_VARIANT="$R9V_TIERED_IQ_MOE_VARIANT" \
    --env QWEN38_TIERED_PREFILL_GROUP_SIZE="$R9V_TIERED_PREFILL_GROUP_SIZE" \
    --env QWEN38_TIERED_EXPERT_CACHE_SLOTS="$R9V_TIERED_EXPERT_CACHE_SLOTS" \
    --env QWEN38_TIERED_EXPERT_CACHE_RANKS="$R9V_TIERED_EXPERT_CACHE_RANKS" \
    --env QWEN38_TIERED_EXPERT_CACHE_POLICY="$R9V_TIERED_EXPERT_CACHE_POLICY" \
    --env QWEN38_TIERED_EXPERT_CACHE_ASYNC="$R9V_TIERED_EXPERT_CACHE_ASYNC" \
    --env QWEN38_TIERED_STREAM_COMPACTION="${R9V_STREAM_EXPERT_COMPACTION:-0}" \
    --env QWEN38_USE_DENSE_MMVQ_HIP=1 \
    --env QWEN38_USE_DENSE_MMVQ_REUSE2=1 \
    --env QWEN38_USE_DENSE_MMVQ_Q8_REUSE2=1 \
    --env QWEN38_USE_DENSE_MMVQ_REUSE3=1 \
    --env QWEN38_USE_DENSE_MMVQ_REUSE4=0 \
    --env QWEN38_USE_DENSE_MMVQ_Q8_ATTN_M3="$R9V_ENABLE_DENSE_Q8_ATTN_M3" \
    --env QWEN38_DENSE_MMVQ_Q8_ATTN_M3_VARIANT="$R9V_DENSE_Q8_ATTN_M3_VARIANT" \
    --env QWEN38_USE_DENSE_HC_DOWN_BF16_M3="$R9V_ENABLE_DENSE_HC_DOWN_BF16_M3" \
    --env QWEN38_FUSED_HC_UP_MIX="$R9V_ENABLE_FUSED_HC_UP_MIX" \
    --env VLLM_GGUF_FUSED_MOE_SHARED_EPILOGUE="$R9V_ENABLE_FUSED_MOE_SHARED_EPILOGUE" \
    --env QWEN38_USE_HIP_FUSED_GDN_MTP="$R9V_ENABLE_FUSED_GDN_MTP" \
    --env VLLM_QWEN4_EXP_RDNA4_QSA_STRIDED="$R9V_ENABLE_RDNA4_QSA_STRIDED" \
    --env VLLM_GGUF_NATIVE_SAFE_MOE_IDS=1 \
    --env VLLM_GGUF_QWEN4_EXP_MULTIMODAL=1 \
    --env VLLM_QWEN4_EXP_MTP_FP8_EXPERT_ONLY=0 \
    --env VLLM_QWEN4_EXP_MTP_FUSED_FC_GATHER=0 \
    --env VLLM_KV_CACHE_LAYOUT=BLHNC \
    --env VLLM_ROCM_MOE_PADDING=0 \
    --env NCCL_ALGO=Ring \
    --env NCCL_PROTO=Simple \
    --env VLLM_ROCM_USE_AITER=1 \
    --env VLLM_ROCM_USE_AITER_LINEAR=0 \
    --env VLLM_ROCM_USE_AITER_MHA=0 \
    --env VLLM_ROCM_USE_AITER_MLA=0 \
    --env VLLM_ROCM_USE_AITER_MOE=0 \
    --env VLLM_ROCM_USE_AITER_RMSNORM=0 \
    --env VLLM_ROCM_USE_AITER_FP8BMM=0 \
    --env VLLM_ROCM_USE_AITER_FP4BMM=0 \
    --env VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1 \
    --env VLLM_PLE_CPU_OFFLOAD=1 \
    --env R9V_PLE_HOST_FENCE="$R9V_PLE_HOST_FENCE" \
    --env VLLM_PLE_RESIDENCY_MODE="$R9V_PLE_RESIDENCY_MODE" \
    --env VLLM_PLE_MMAP_HOST_REGISTER="$ple_mmap_host_register" \
    --env VLLM_PLE_MMAP_HOST_REGISTER_EXPECTED_BYTES=28800138240 \
    --env VLLM_PLE_PINNED_RESERVE_BYTES="$R9V_PLE_PINNED_RESERVE_BYTES" \
    --env VLLM_PLE_BOUNDED_BYTES=4294967296 \
    --env VLLM_PLE_BOUNDED_CHUNK_BYTES=4096 \
    --env VLLM_PLE_MMAP_READAHEAD="$R9V_PLE_MMAP_READAHEAD" \
    --env VLLM_PLE_RSS_LOG_ROWS="$R9V_PLE_RSS_LOG_ROWS" \
    --env VLLM_PLE_WORKER_TIMING="$R9V_PLE_WORKER_TIMING" \
    --env R9V_WORKER_DIAGNOSTICS=1 \
    --env R9V_CONTAINER_NAME="$container" \
    --env R9V_OBSERVABILITY_TARGET="${R9V_OBSERVABILITY_TARGET:-}" \
    --env R9V_STAGE_DIAGNOSTICS="${R9V_STAGE_DIAGNOSTICS:-0}" \
    --env R9V_EXPECTED_GPU_BDFS="${R9V_EXPECTED_GPU_BDFS:-}" \
    --env VLLM_CUSTOM_SCOPES_FOR_PROFILING="$custom_scopes" \
    --env QWEN38_PROFILE_DENSE_SHAPES="$dense_scopes" \
    --env GGUF_PLE_MMAP_PATH=/ple/per_layer_token_embd.iq4_nl.bin \
    --env GGUF_PLE_MMAP_TRIM_ROWS="$R9V_PLE_MMAP_TRIM_ROWS" \
    "$image" \
    "/models/$target_rel" \
    --tokenizer "/models/$metadata_rel" \
    --hf-config-path "/models/$metadata_rel" \
    --served-model-name "$R9V_SERVED_MODEL_NAME" \
    "${eager_args[@]}" \
    "${prefix_cache_args[@]}" \
    --load-format gguf \
    --quantization gguf \
    --tensor-parallel-size "$R9V_TENSOR_PARALLEL_SIZE" \
    --pipeline-parallel-size 1 \
    --cpu-offload-gb "$R9V_CPU_OFFLOAD_GB" \
    --cpu-offload-params experts \
    --kv-cache-memory-bytes "$R9V_KV_CACHE_MEMORY_BYTES" \
    "${speculative_args[@]}" \
    --max-model-len "$R9V_MAX_MODEL_LEN" \
    --max-num-seqs "$R9V_MAX_NUM_SEQS" \
    --max-num-batched-tokens "$R9V_MAX_NUM_BATCHED_TOKENS" \
    --compilation-config "$compilation_config" \
    --model-loader-extra-config "{\"mm_proj\":\"/models/$mmproj_rel\"}" \
    --limit-mm-per-prompt '{"image":1,"video":0}' \
    --mm-processor-kwargs '{"min_pixels":65536,"max_pixels":262144}' \
    --mm-processor-cache-gb 0 \
    --mm-encoder-tp-mode weights \
    "${auto_tool_args[@]}" \
    --tool-call-parser "$R9V_TOOL_CALL_PARSER" \
    --reasoning-parser "$R9V_REASONING_PARSER" \
    "${profiler_args[@]}" \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000

printf 'Started %s; health endpoint: http://127.0.0.1:%s/health\n' \
    "$container" "$R9V_HOST_PORT"

printf 'Logs are retained with the container (rotating 5 x 20 MiB). Before removing it, run: ./r9v support %s --output <new-directory>\n' \
    "${R9V_PROFILE_ID:-qwen38}"
