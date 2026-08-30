#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
profile=${R9V_PROFILE:-$repo_root/profiles/qwen38-flash-next/dual-r9700/profile.env}
[[ -r "$profile" ]] || { printf 'Profile not found: %s\n' "$profile" >&2; exit 1; }
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

for value in \
    R9V_MTP_LOCAL_ARGMAX \
    R9V_ENABLE_AUTO_TOOL_CHOICE \
    R9V_R4D \
    R9V_R4D_AR \
    R9V_R4D_GDN \
    R9V_R4D_AR_QUANT; do
    [[ ${!value} == 0 || ${!value} == 1 ]] || {
        printf '%s must be 0 or 1\n' "$value" >&2
        exit 2
    }
done
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
metadata_rel=${R9V_METADATA_REL:-metadata}
mtp_rel=${R9V_MTP_REL:-mtp}
mmproj_rel=${R9V_MMPROJ_REL:-vision/mmproj-Qwen3.8-Flash-Next-Q8_0.gguf}
manifest_rel=${R9V_MANIFEST_REL:-manifests/hot-manifest-q4-vision-128k-multiprompt-r1-lru16-neutral.json}

for required in \
    "$model_dir/$target_rel" \
    "$model_dir/$target_shard2_rel" \
    "$model_dir/$target_shard3_rel" \
    "$model_dir/$metadata_rel/config.json" \
    "$model_dir/$mtp_rel/config.json" \
    "$model_dir/$mtp_rel/model.safetensors" \
    "$model_dir/$mmproj_rel" \
    "$model_dir/$manifest_rel" \
    "$ple_path"; do
    [[ -f "$required" ]] || { printf 'Required file missing: %s\n' "$required" >&2; exit 1; }
done

if docker container inspect "$container" >/dev/null 2>&1; then
    printf 'Container already exists: %s\nStop/remove it explicitly before relaunch.\n' \
        "$container" >&2
    exit 1
fi

mkdir -p "$cache_dir"
render_gid=$(getent group render | cut -d: -f3)
video_gid=$(getent group video | cut -d: -f3)

docker run --detach \
    --name "$container" \
    --device /dev/kfd \
    --device /dev/dri \
    --group-add "$render_gid" \
    --group-add "$video_gid" \
    --ipc host \
    --security-opt seccomp=unconfined \
    --security-opt label=disable \
    --publish "$R9V_HOST_PORT:8000" \
    --volume "$model_dir:/models:ro" \
    --volume "$ple_path:/ple/per_layer_token_embd.iq4_nl.bin:ro" \
    --volume "$cache_dir:/cache" \
    --env HIP_VISIBLE_DEVICES="$visible_devices" \
    --env ROCR_VISIBLE_DEVICES="$visible_devices" \
    --env VLLM_CACHE_ROOT=/cache/vllm \
    --env RADIANCE_CPU_OFFLOAD_GB_BY_DEVICE="$R9V_CPU_OFFLOAD_GB_BY_DEVICE" \
    --env R9V_CPU_OFFLOAD_GB_BY_DEVICE="$R9V_CPU_OFFLOAD_GB_BY_DEVICE" \
    --env RADIANCE_TIERED_EXPERT_MANIFEST="/models/$manifest_rel" \
    --env RADIANCE_UVA_HOST_COHERENCE=default \
    --env RADIANCE_UVA_HOST_NONCOHERENT=0 \
    --env RADIANCE_USE_R4D=0 \
    --env RADIANCE_USE_R4D_AR=0 \
    --env RADIANCE_USE_R4D_GDN=0 \
    --env RADIANCE_USE_R4D_AR_QUANT=0 \
    --env QWEN38_USE_TIERED_IQ_MOE_HIP=1 \
    --env QWEN38_TIERED_IQ_MOE_VARIANT="$R9V_TIERED_IQ_MOE_VARIANT" \
    --env QWEN38_TIERED_EXPERT_CACHE_SLOTS="$R9V_TIERED_EXPERT_CACHE_SLOTS" \
    --env QWEN38_TIERED_EXPERT_CACHE_RANKS="$R9V_TIERED_EXPERT_CACHE_RANKS" \
    --env QWEN38_TIERED_EXPERT_CACHE_POLICY="$R9V_TIERED_EXPERT_CACHE_POLICY" \
    --env QWEN38_TIERED_EXPERT_CACHE_ASYNC="$R9V_TIERED_EXPERT_CACHE_ASYNC" \
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
    --env VLLM_PLE_RESIDENCY_MODE="$R9V_PLE_RESIDENCY_MODE" \
    --env VLLM_PLE_MMAP_HOST_REGISTER=0 \
    --env VLLM_PLE_MMAP_HOST_REGISTER_EXPECTED_BYTES=28800138240 \
    --env VLLM_PLE_BOUNDED_BYTES=4294967296 \
    --env VLLM_PLE_BOUNDED_CHUNK_BYTES=4096 \
    --env VLLM_PLE_RSS_LOG_ROWS="$R9V_PLE_RSS_LOG_ROWS" \
    --env GGUF_PLE_MMAP_PATH=/ple/per_layer_token_embd.iq4_nl.bin \
    --env GGUF_PLE_MMAP_TRIM_ROWS="$R9V_PLE_MMAP_TRIM_ROWS" \
    "$image" \
    "/models/$target_rel" \
    --tokenizer "/models/$metadata_rel" \
    --hf-config-path "/models/$metadata_rel" \
    --served-model-name "$R9V_SERVED_MODEL_NAME" \
    --load-format gguf \
    --quantization gguf \
    --tensor-parallel-size "$R9V_TENSOR_PARALLEL_SIZE" \
    --pipeline-parallel-size 1 \
    --cpu-offload-gb "$R9V_CPU_OFFLOAD_GB" \
    --cpu-offload-params experts \
    --kv-cache-memory-bytes "$R9V_KV_CACHE_MEMORY_BYTES" \
    --speculative-config "{\"method\":\"mtp\",\"model\":\"/models/$mtp_rel\",\"num_speculative_tokens\":$R9V_MTP_SPEC_TOKENS,\"draft_tensor_parallel_size\":$R9V_MTP_DRAFT_TP_SIZE,\"quantization\":\"$R9V_MTP_QUANTIZATION\",\"use_local_argmax_reduction\":$local_argmax,\"draft_load_config\":{\"load_format\":\"auto\"}}" \
    --max-model-len "$R9V_MAX_MODEL_LEN" \
    --max-num-seqs "$R9V_MAX_NUM_SEQS" \
    --max-num-batched-tokens "$R9V_MAX_NUM_BATCHED_TOKENS" \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,3],"max_cudagraph_capture_size":3}' \
    --model-loader-extra-config "{\"mm_proj\":\"/models/$mmproj_rel\"}" \
    --limit-mm-per-prompt '{"image":1,"video":0}' \
    --mm-processor-kwargs '{"min_pixels":65536,"max_pixels":262144}' \
    --mm-processor-cache-gb 0 \
    --mm-encoder-tp-mode weights \
    "${auto_tool_args[@]}" \
    --tool-call-parser "$R9V_TOOL_CALL_PARSER" \
    --reasoning-parser "$R9V_REASONING_PARSER" \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000

printf 'Started %s; health endpoint: http://127.0.0.1:%s/health\n' \
    "$container" "$R9V_HOST_PORT"
