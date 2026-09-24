# R9V modification: Qwen3.8 Flash Next ROCm integration and profiling support.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernels for the Qwen4Exp weight-free QSA path."""

from __future__ import annotations

import math
import os

import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.platforms.rocm import on_rdna4
from vllm.triton_utils import HAS_TRITON, tl, triton

_LOGITS_WORKSPACE_BYTES = 128 * 1024 * 1024
_TOPK_WORKSPACE_BYTES = 1024 * 1024
_QSA_MQA_BLOCK_N = 32
_RDNA4_QSA_STRIDED_ENV = "VLLM_QWEN4_EXP_RDNA4_QSA_STRIDED"
_RDNA4_QSA_STRIDED_PROGRAMS = 128


@triton.jit
def _qsa_mqa_paged_kernel(
    q_ptr,
    k_cache_ptr,
    page_table_ptr,
    token_to_req_ptr,
    query_positions_ptr,
    sequence_lengths_ptr,
    visible_blocks_ptr,
    logits_ptr,
    stride_q_row,
    stride_q_head,
    stride_q_dim,
    stride_cache_block,
    stride_cache_token,
    stride_cache_dim,
    stride_table_req,
    stride_table_page,
    stride_logits_row,
    num_rows,
    num_columns,
    num_pages,
    num_requests,
    score_divisor,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    dims = tl.arange(0, BLOCK_D)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    query_position = tl.load(query_positions_ptr + row)
    sequence_length = tl.load(
        sequence_lengths_ptr + safe_request,
        mask=(request >= 0) & (request < num_requests),
        other=0,
    )
    visible = tl.minimum(
        (query_position + 1) // COMPRESS_RATIO,
        sequence_length // COMPRESS_RATIO,
    )
    if tl.program_id(1) == 0:
        tl.store(visible_blocks_ptr + row, visible)
    logical_page = columns // PAGE_SIZE
    page_offset = columns % PAGE_SIZE
    valid = (
        (row < num_rows)
        & (columns < num_columns)
        & (columns < visible)
        & (request >= 0)
        & (request < num_requests)
        & (logical_page < PAGE_TABLE_WIDTH)
    )
    safe_logical_page = tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1)
    physical_page = tl.load(
        page_table_ptr
        + safe_request * stride_table_req
        + safe_logical_page * stride_table_page,
        mask=valid,
        other=-1,
    )
    valid &= (physical_page >= 0) & (physical_page < num_pages)
    # physical_page * block stride can overflow int32 for large caches.
    safe_physical_page = tl.maximum(physical_page, 0).to(tl.int64)
    score = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for head in tl.static_range(0, NUM_HEADS):
        query = tl.load(
            q_ptr + row * stride_q_row + head * stride_q_head + dims * stride_q_dim,
            mask=dims < HEAD_DIM,
            other=0.0,
        ).to(tl.float32)
        keys = tl.load(
            k_cache_ptr
            + safe_physical_page[:, None] * stride_cache_block
            + page_offset[:, None] * stride_cache_token
            + dims[None, :] * stride_cache_dim,
            mask=valid[:, None] & (dims[None, :] < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        dot = tl.sum(keys * query[None, :], axis=1)
        score += tl.maximum(dot, 0.0)

    score /= score_divisor
    tl.store(
        logits_ptr + row * stride_logits_row + columns,
        tl.where(valid, score, -float("inf")),
        mask=(row < num_rows) & (columns < num_columns),
    )


@triton.jit
def _qsa_mqa_paged_strided_kernel(
    q_ptr,
    k_cache_ptr,
    page_table_ptr,
    token_to_req_ptr,
    query_positions_ptr,
    sequence_lengths_ptr,
    visible_blocks_ptr,
    logits_ptr,
    stride_q_row,
    stride_q_head,
    stride_q_dim,
    stride_cache_block,
    stride_cache_token,
    stride_cache_dim,
    stride_table_req,
    stride_table_page,
    stride_logits_row,
    num_rows,
    num_columns,
    num_pages,
    num_requests,
    score_divisor,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PROGRAM_STRIDE: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
) -> None:
    """Score live QSA tiles over a fixed, strided scalar program grid."""

    row = tl.program_id(0)
    first_tile = tl.program_id(1)
    dims = tl.arange(0, BLOCK_D)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    query_position = tl.load(query_positions_ptr + row)
    sequence_length = tl.load(
        sequence_lengths_ptr + safe_request,
        mask=(request >= 0) & (request < num_requests),
        other=0,
    )
    visible = tl.minimum(
        (query_position + 1) // COMPRESS_RATIO,
        sequence_length // COMPRESS_RATIO,
    )
    if first_tile == 0:
        tl.store(visible_blocks_ptr + row, visible)
    tile_end = tl.minimum(tl.cdiv(visible, BLOCK_N), tl.cdiv(num_columns, BLOCK_N))
    if first_tile >= tile_end:
        return

    column_offsets = tl.arange(0, BLOCK_N)
    for tile in tl.range(first_tile, tile_end, PROGRAM_STRIDE):
        columns = tile * BLOCK_N + column_offsets
        logical_page = columns // PAGE_SIZE
        page_offset = columns % PAGE_SIZE
        valid = (
            (row < num_rows)
            & (columns < num_columns)
            & (columns < visible)
            & (request >= 0)
            & (request < num_requests)
            & (logical_page < PAGE_TABLE_WIDTH)
        )
        safe_logical_page = tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1)
        physical_page = tl.load(
            page_table_ptr
            + safe_request * stride_table_req
            + safe_logical_page * stride_table_page,
            mask=valid,
            other=-1,
        )
        valid &= (physical_page >= 0) & (physical_page < num_pages)
        # physical_page * block stride can overflow int32 for large caches.
        safe_physical_page = tl.maximum(physical_page, 0).to(tl.int64)
        score = tl.zeros((BLOCK_N,), dtype=tl.float32)

        for head in tl.static_range(0, NUM_HEADS):
            query = tl.load(
                q_ptr
                + row * stride_q_row
                + head * stride_q_head
                + dims * stride_q_dim,
                mask=dims < HEAD_DIM,
                other=0.0,
            ).to(tl.float32)
            keys = tl.load(
                k_cache_ptr
                + safe_physical_page[:, None] * stride_cache_block
                + page_offset[:, None] * stride_cache_token
                + dims[None, :] * stride_cache_dim,
                mask=valid[:, None] & (dims[None, :] < HEAD_DIM),
                other=0.0,
            ).to(tl.float32)
            dot = tl.sum(keys * query[None, :], axis=1)
            score += tl.maximum(dot, 0.0)

        score /= score_divisor
        tl.store(
            logits_ptr + row * stride_logits_row + columns,
            tl.where(valid, score, -float("inf")),
            mask=(row < num_rows) & (columns < num_columns),
        )


@triton.jit
def _expand_qsa_indices_kernel(
    block_indices_ptr,
    query_positions_ptr,
    sequence_lengths_ptr,
    token_to_req_ptr,
    output_ptr,
    stride_blocks_row,
    stride_blocks_column,
    stride_output_row,
    stride_output_column,
    rows,
    num_requests,
    BLOCK_TOPK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    TOKEN_TOPK: tl.constexpr,
    OUTPUT_WIDTH: tl.constexpr,
    COLUMN_BLOCK: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    columns = tl.program_id(1) * COLUMN_BLOCK + tl.arange(0, COLUMN_BLOCK)
    query_position = tl.load(query_positions_ptr + row)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    sequence_length = tl.load(
        sequence_lengths_ptr + safe_request,
        mask=(request >= 0) & (request < num_requests),
        other=0,
    )
    complete_blocks = tl.minimum(
        tl.minimum(
            (query_position + 1) // COMPRESS_RATIO,
            sequence_length // COMPRESS_RATIO,
        ),
        BLOCK_TOPK,
    )
    expanded_count = complete_blocks * COMPRESS_RATIO
    tail_start = ((query_position + 1) // COMPRESS_RATIO) * COMPRESS_RATIO
    tail_count = (query_position + 1) - tail_start

    is_expanded = columns < expanded_count
    block_rank = columns // COMPRESS_RATIO
    offset = columns % COMPRESS_RATIO
    safe_rank = tl.minimum(block_rank, BLOCK_TOPK - 1)
    block = tl.load(
        block_indices_ptr + row * stride_blocks_row + safe_rank * stride_blocks_column,
        mask=(row < rows) & is_expanded,
        other=-1,
    )
    expanded = block * COMPRESS_RATIO + offset
    tail_offset = columns - expanded_count
    is_tail = (
        (columns >= expanded_count)
        & (tail_offset < tail_count)
        & (tail_offset < COMPRESS_RATIO - 1)
    )
    token = tl.where(is_expanded, expanded, tail_start + tail_offset)
    valid = (
        (row < rows)
        & (columns < OUTPUT_WIDTH)
        & (is_expanded | is_tail)
        & (token >= 0)
        & (token < sequence_length)
    )
    tl.store(
        output_ptr + row * stride_output_row + columns * stride_output_column,
        tl.where(valid, token, -1),
        mask=(row < rows) & (columns < OUTPUT_WIDTH),
    )


@triton.jit
def _qsa_sparse_paged_gqa_splitk_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    indices_ptr,
    block_table_ptr,
    token_to_req_ptr,
    partial_output_ptr,
    partial_lse_ptr,
    output_ptr,
    stride_q_row,
    stride_q_head,
    stride_k_block,
    stride_k_token,
    stride_k_head,
    stride_v_block,
    stride_v_token,
    stride_v_head,
    stride_indices_row,
    stride_table_req,
    stride_output_row,
    stride_output_head,
    num_rows,
    num_cache_blocks,
    num_requests,
    TOPK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    FP32_PV: tl.constexpr = False,
) -> None:
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    split_id = tl.program_id(2)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)

    head_offsets = tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, HEAD_DIM)
    column_offsets = tl.arange(0, BLOCK_N)
    first_head = kv_head * GROUP_SIZE
    query = tl.load(
        q_ptr
        + row * stride_q_row
        + (first_head + head_offsets[:, None]) * stride_q_head
        + dim_offsets[None, :],
        mask=head_offsets[:, None] < GROUP_SIZE,
        other=0.0,
    )

    max_value = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    normalizer = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    softmax_scale_log2: tl.constexpr = (HEAD_DIM**-0.5) * 1.4426950408889634

    # Dynamic bounds avoid padded main-loop iterations for uneven splits.
    split_tile_start = split_id * NUM_TILES // NUM_SPLITS
    split_tile_end = (split_id + 1) * NUM_TILES // NUM_SPLITS
    for tile in range(split_tile_start, split_tile_end):
        columns = tile * BLOCK_N + column_offsets
        logical_token = tl.load(
            indices_ptr + row * stride_indices_row + columns,
            mask=columns < TOPK,
            other=-1,
        )
        safe_token = tl.maximum(logical_token, 0)
        logical_page = safe_token // PAGE_SIZE
        page_offset = safe_token % PAGE_SIZE
        valid = (
            (request >= 0)
            & (request < num_requests)
            & (logical_token >= 0)
            & (logical_page < PAGE_TABLE_WIDTH)
        )
        physical_page = tl.load(
            block_table_ptr
            + safe_request * stride_table_req
            + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1),
            mask=valid,
            other=-1,
        )
        valid &= (physical_page >= 0) & (physical_page < num_cache_blocks)
        # physical_page * block stride can overflow int32 for large caches.
        safe_page = tl.maximum(physical_page, 0).to(tl.int64)
        keys = tl.load(
            k_cache_ptr
            + safe_page[None, :] * stride_k_block
            + page_offset[None, :] * stride_k_token
            + kv_head * stride_k_head
            + dim_offsets[:, None],
            mask=valid[None, :],
            other=0.0,
        )
        values = tl.load(
            v_cache_ptr
            + safe_page[:, None] * stride_v_block
            + page_offset[:, None] * stride_v_token
            + kv_head * stride_v_head
            + dim_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        scores = tl.dot(query, keys)
        # Scaling scores avoids re-quantizing a scaled query to BF16.
        scores *= softmax_scale_log2
        scores = tl.where(valid[None, :], scores, -1.0e20)
        next_max = tl.maximum(max_value, tl.max(scores, axis=1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.where(
            valid[None, :], tl.math.exp2(scores - next_max[:, None]), 0.0
        )
        if FP32_PV:
            # R9V: keep probabilities in fp32 for the P.V product (research option);
            # the default rounds them to bf16 relative to the running max, so the
            # rounding depends on the order the selected tokens are visited in.
            accumulator = tl.dot(
                probabilities,
                values.to(tl.float32),
                acc=accumulator * alpha[:, None],
                input_precision="ieee",
            )
        else:
            accumulator = tl.dot(
                probabilities.to(values.dtype),
                values,
                acc=accumulator * alpha[:, None],
            )
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        max_value = next_max

    has_values = normalizer > 0
    normalized_output = tl.where(
        has_values[:, None],
        accumulator / tl.maximum(normalizer[:, None], 1.0e-20),
        0.0,
    )
    output_mask = head_offsets[:, None] < GROUP_SIZE
    if NUM_SPLITS == 1:
        tl.store(
            output_ptr
            + row * stride_output_row
            + (first_head + head_offsets[:, None]) * stride_output_head
            + dim_offsets[None, :],
            normalized_output,
            mask=output_mask,
        )
    else:
        partial_lse = tl.where(
            has_values,
            max_value + tl.math.log2(tl.maximum(normalizer, 1.0e-20)),
            -float("inf"),
        )
        tl.store(
            partial_output_ptr
            + (
                (split_id * num_rows + row) * NUM_QUERY_HEADS
                + first_head
                + head_offsets[:, None]
            )
            * HEAD_DIM
            + dim_offsets[None, :],
            normalized_output,
            mask=output_mask,
        )
        tl.store(
            partial_lse_ptr
            + (split_id * num_rows + row) * NUM_QUERY_HEADS
            + first_head
            + head_offsets,
            partial_lse,
            mask=head_offsets < GROUP_SIZE,
        )


@triton.jit
def _qsa_merge_splitk_kernel(
    partial_output_ptr,
    partial_lse_ptr,
    output_ptr,
    stride_output_row,
    stride_output_head,
    num_rows,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    head = tl.program_id(1)
    split_offsets = tl.arange(0, BLOCK_SPLITS)
    dim_offsets = tl.arange(0, HEAD_DIM)
    split_mask = split_offsets < NUM_SPLITS
    lse = tl.load(
        partial_lse_ptr + (split_offsets * num_rows + row) * NUM_QUERY_HEADS + head,
        mask=split_mask,
        other=-float("inf"),
    )
    lse_max = tl.max(lse, axis=0)
    has_values = lse_max > -float("inf")
    shifted = tl.where(split_mask & has_values, lse - lse_max, -float("inf"))
    weights = tl.math.exp2(shifted)
    denominator = tl.sum(weights, axis=0)
    partial_output = tl.load(
        partial_output_ptr
        + ((split_offsets[:, None] * num_rows + row) * NUM_QUERY_HEADS + head)
        * HEAD_DIM
        + dim_offsets[None, :],
        mask=split_mask[:, None],
        other=0.0,
    )
    merged = tl.sum(partial_output * weights[:, None], axis=0)
    merged = tl.where(denominator > 0, merged / denominator, 0.0)
    tl.store(
        output_ptr + row * stride_output_row + head * stride_output_head + dim_offsets,
        merged,
    )


@triton.jit
def _store_qsa_rows_kernel(
    cache_ptr,
    slots_ptr,
    rows_ptr,
    stride_cache_block,
    stride_cache_token,
    stride_cache_dim,
    stride_rows_row,
    stride_rows_dim,
    num_rows,
    num_blocks,
    PAGE_SIZE: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    slot = tl.load(slots_ptr + row)
    valid = (row < num_rows) & (slot >= 0) & (slot < num_blocks * PAGE_SIZE)
    block = tl.maximum(slot, 0) // PAGE_SIZE
    token = tl.maximum(slot, 0) % PAGE_SIZE
    values = tl.load(
        rows_ptr + row * stride_rows_row + dims * stride_rows_dim,
        mask=valid & (dims < WIDTH),
        other=0,
    )
    tl.store(
        cache_ptr
        + block * stride_cache_block
        + token * stride_cache_token
        + dims * stride_cache_dim,
        values,
        mask=valid & (dims < WIDTH),
    )


@triton.jit
def _compress_qsa_groups_kernel(
    raw_keys_ptr,  # this step's raw key rows, straight from activations
    raw_positions_ptr,  # this step's per-token positions
    compressor_state_cache_ptr,  # per-request ring of previous raw keys
    rope_cache_ptr,  # packed RoPE position tail of the ring
    compressor_state_table_ptr,
    token_to_req_ptr,
    query_start_loc_ptr,
    logical_positions_ptr,
    compressed_slots_ptr,
    pooled_ptr,
    first_positions_ptr,
    stride_raw_row,
    stride_raw_dim,
    stride_raw_positions_row,
    stride_raw_positions_dim,
    stride_compressor_state_block,
    stride_compressor_state_token,
    stride_compressor_state_dim,
    stride_rope_block,
    stride_rope_token,
    stride_rope_dim,
    stride_compressor_state_table_req,
    stride_pooled_row,
    stride_pooled_dim,
    stride_positions_row,
    stride_positions_dim,
    num_rows,
    num_compressor_state_blocks,
    num_requests,
    COMPRESSOR_STATE_SIZE: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    LOAD_ROPE_POSITIONS: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    request = tl.load(token_to_req_ptr + row)
    end_position = tl.load(logical_positions_ptr + row)
    compressed_slot = tl.load(compressed_slots_ptr + row)
    valid_request = (request >= 0) & (request < num_requests)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    query_row_start = tl.load(
        query_start_loc_ptr + safe_request, mask=valid_request, other=0
    )
    query_row_end = tl.load(
        query_start_loc_ptr + safe_request + 1, mask=valid_request, other=0
    )
    chunk_start_position = end_position - (row - query_row_start)
    compressor_state_block = tl.load(
        compressor_state_table_ptr + safe_request * stride_compressor_state_table_req,
        mask=valid_request,
        other=-1,
    )
    valid_compressor_state_block = (compressor_state_block >= 0) & (
        compressor_state_block < num_compressor_state_blocks
    )
    valid_row = (
        (row < num_rows)
        & valid_request
        & (row >= query_row_start)
        & (row < query_row_end)
        & (end_position >= COMPRESS_RATIO - 1)
        & (compressed_slot >= 0)
    )
    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)

    # A group can span the compressor-state ring (older members) and this
    # step's raw rows (members at positions >= chunk_start_position).
    for group_offset in tl.range(0, COMPRESS_RATIO):
        position = end_position - (COMPRESS_RATIO - 1 - group_offset)
        use_raw = position >= chunk_start_position
        raw_row = query_row_start + position - chunk_start_position
        raw_values = tl.load(
            raw_keys_ptr + raw_row * stride_raw_row + dims * stride_raw_dim,
            mask=valid_row
            & use_raw
            & (raw_row >= query_row_start)
            & (raw_row < query_row_end)
            & (raw_row < num_rows)
            & (dims < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        compressor_state_values = tl.load(
            compressor_state_cache_ptr
            + tl.maximum(compressor_state_block, 0).to(tl.int64)
            * stride_compressor_state_block
            + (position % COMPRESSOR_STATE_SIZE) * stride_compressor_state_token
            + dims * stride_compressor_state_dim,
            mask=valid_row
            & ~use_raw
            & valid_compressor_state_block
            & (dims < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        accumulator += tl.where(use_raw, raw_values, compressor_state_values)

    tl.store(
        pooled_ptr + row * stride_pooled_row + dims * stride_pooled_dim,
        accumulator / COMPRESS_RATIO,
        mask=(row < num_rows) & (dims < HEAD_DIM),
    )

    position_dims = tl.arange(0, 4)
    first_position = end_position - COMPRESS_RATIO + 1
    if LOAD_ROPE_POSITIONS:
        first_from_raw = first_position >= chunk_start_position
        raw_first_row = query_row_start + first_position - chunk_start_position
        raw_position_values = tl.load(
            raw_positions_ptr
            + raw_first_row * stride_raw_positions_row
            + position_dims * stride_raw_positions_dim,
            mask=valid_row
            & first_from_raw
            & (raw_first_row >= query_row_start)
            & (raw_first_row < query_row_end)
            & (raw_first_row < num_rows)
            & (position_dims < 3),
            other=0,
        )
        compressor_state_position_values = tl.load(
            rope_cache_ptr
            + tl.maximum(compressor_state_block, 0).to(tl.int64) * stride_rope_block
            + (first_position % COMPRESSOR_STATE_SIZE) * stride_rope_token
            + position_dims * stride_rope_dim,
            mask=valid_row
            & ~first_from_raw
            & valid_compressor_state_block
            & (position_dims < 3),
            other=0,
        )
        position_values = tl.where(
            first_from_raw,
            raw_position_values,
            compressor_state_position_values,
        )
    else:
        position_values = tl.where(valid_row, first_position, 0)
    tl.store(
        first_positions_ptr
        + row * stride_positions_row
        + position_dims * stride_positions_dim,
        position_values,
        mask=(row < num_rows) & (position_dims < 3),
    )


def _validate_mqa(q: torch.Tensor) -> None:
    if q.ndim != 3 or q.shape[1] <= 0 or q.shape[2] <= 0:
        raise ValueError("QSA query must be [rows, heads, head_dim]")


def _rdna4_qsa_strided_enabled() -> bool:
    """Return whether the bounded strided scalar scorer is enabled."""

    return os.environ.get(_RDNA4_QSA_STRIDED_ENV, "0") == "1" and on_rdna4()


def _rdna4_qsa_strided_grid(num_rows: int, num_columns: int) -> tuple[int, int]:
    num_tiles = (num_columns + _QSA_MQA_BLOCK_N - 1) // _QSA_MQA_BLOCK_N
    return num_rows, min(num_tiles, _RDNA4_QSA_STRIDED_PROGRAMS)


def qsa_mqa_paged(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    compress_ratio: int,
    num_columns: int | None = None,
    score_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute QSA scores directly from a paged compressed-key cache."""

    _validate_mqa(q)
    if not q.is_cuda or not HAS_TRITON:
        raise RuntimeError("paged QSA scoring requires a GPU and Triton")
    if k_cache.ndim != 4 or k_cache.shape[2] != 1:
        raise ValueError("QSA cache must be [pages, page_size, 1, head_dim]")
    if k_cache.shape[3] != q.shape[2]:
        raise ValueError("QSA query and cache dimensions must match")
    if page_table.ndim != 2:
        raise ValueError("QSA page table must be two-dimensional")
    if q.shape[0] and (not all(k_cache.shape[:2]) or not all(page_table.shape)):
        raise ValueError("QSA paged scoring cache and page table must be nonempty")
    if token_to_req.shape != (q.shape[0],):
        raise ValueError("QSA request mapping must match query rows")
    if query_positions.shape != (q.shape[0],):
        raise ValueError("QSA query positions must match query rows")
    if sequence_lengths.shape != (page_table.shape[0],):
        raise ValueError("QSA sequence lengths must match page-table requests")
    if compress_ratio <= 0:
        raise ValueError("QSA compression ratio must be positive")
    score_divisor = math.sqrt(q.shape[2]) if score_scale is None else score_scale
    if score_divisor <= 0:
        raise ValueError("QSA score scale must be positive")

    capacity = page_table.shape[1] * k_cache.shape[1]
    columns = capacity if num_columns is None else num_columns
    if columns < 0:
        raise ValueError("QSA score width must be non-negative")
    logits = torch.empty((q.shape[0], columns), dtype=torch.float32, device=q.device)
    visible_blocks = torch.empty(q.shape[0], dtype=torch.int32, device=q.device)
    if not q.shape[0] or not columns:
        return logits, visible_blocks
    block_n = _QSA_MQA_BLOCK_N
    if _rdna4_qsa_strided_enabled():
        grid = _rdna4_qsa_strided_grid(q.shape[0], columns)
        _qsa_mqa_paged_strided_kernel[grid](
            q,
            k_cache,
            page_table,
            token_to_req,
            query_positions,
            sequence_lengths,
            visible_blocks,
            logits,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(3),
            page_table.stride(0),
            page_table.stride(1),
            logits.stride(0),
            q.shape[0],
            columns,
            k_cache.shape[0],
            page_table.shape[0],
            float(score_divisor),
            PAGE_SIZE=k_cache.shape[1],
            PAGE_TABLE_WIDTH=page_table.shape[1],
            NUM_HEADS=q.shape[1],
            HEAD_DIM=q.shape[2],
            BLOCK_N=block_n,
            BLOCK_D=triton.next_power_of_2(q.shape[2]),
            PROGRAM_STRIDE=grid[1],
            COMPRESS_RATIO=compress_ratio,
            num_warps=4,
        )
        return logits, visible_blocks
    _qsa_mqa_paged_kernel[(q.shape[0], triton.cdiv(columns, block_n))](
        q,
        k_cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        visible_blocks,
        logits,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(3),
        page_table.stride(0),
        page_table.stride(1),
        logits.stride(0),
        q.shape[0],
        columns,
        k_cache.shape[0],
        page_table.shape[0],
        float(score_divisor),
        PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=page_table.shape[1],
        NUM_HEADS=q.shape[1],
        HEAD_DIM=q.shape[2],
        BLOCK_N=block_n,
        BLOCK_D=triton.next_power_of_2(q.shape[2]),
        COMPRESS_RATIO=compress_ratio,
        num_warps=4,
    )
    return logits, visible_blocks


def expand_qsa_block_indices_cuda(
    block_indices: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    token_to_req: torch.Tensor,
    compress_ratio: int,
    token_topk: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Expand compressed blocks and compact the causal tail of the open group."""

    if not block_indices.is_cuda or not HAS_TRITON:
        raise RuntimeError("QSA index expansion requires a GPU and Triton")
    if token_topk % compress_ratio:
        raise ValueError("QSA token top-k must be divisible by compression ratio")
    block_topk = token_topk // compress_ratio
    output_width = token_topk + compress_ratio - 1
    if block_indices.shape != (query_positions.numel(), block_topk):
        raise ValueError("QSA compressed top-k has an invalid shape")
    if token_to_req.shape != query_positions.shape:
        raise ValueError("QSA request mapping must match query positions")
    if sequence_lengths.ndim != 1 or not sequence_lengths.shape[0]:
        raise ValueError("QSA request sequence lengths must be nonempty")
    if out is None:
        out = torch.empty(
            (block_indices.shape[0], output_width),
            dtype=torch.int32,
            device=block_indices.device,
        )
    elif out.shape != (block_indices.shape[0], output_width):
        raise ValueError("QSA expansion output has an invalid shape")
    if not block_indices.shape[0]:
        return out
    column_block = 256
    _expand_qsa_indices_kernel[
        (block_indices.shape[0], triton.cdiv(output_width, column_block))
    ](
        block_indices,
        query_positions,
        sequence_lengths,
        token_to_req,
        out,
        block_indices.stride(0),
        block_indices.stride(1),
        out.stride(0),
        out.stride(1),
        block_indices.shape[0],
        sequence_lengths.shape[0],
        BLOCK_TOPK=block_topk,
        COMPRESS_RATIO=compress_ratio,
        TOKEN_TOPK=token_topk,
        OUTPUT_WIDTH=output_width,
        COLUMN_BLOCK=column_block,
        num_warps=4,
    )
    return out


# R9V: QSA block top-k selection mode (R9V_QSA_TOPK; R9V_QSA_TOPK_CONTROL names a file
# whose content overrides it per call, for research A/B within one server).
#   ordered (default, the fix): exact top-k, ties to the lower block index, blocks in
#       ascending order. The radix kernel supplies only each row's k-th score; a prefix-sum
#       kernel writes the selection. Deterministic.
#   radix: vLLM's top_k_per_row_decode alone (the image's behaviour). Its set is correct,
#       but tied blocks and the output order follow atomic arrival, so the sparse attention
#       sums in a different order on every run.
#   stable: same result as ordered, computed in torch (reference; slow).
#   stable_recent / ordered_recent: ties toward the most recent block instead.
#   radix_sorted / stable_desc: research only; same sets as radix / stable in another order.
#   check: radix, validated against stable on the same scores, statistics appended to
#       R9V_QSA_TOPK_CHECK_LOG (host syncs; eager prefill only).
_QSA_TOPK_ENV = "R9V_QSA_TOPK"
_QSA_TOPK_CONTROL_ENV = "R9V_QSA_TOPK_CONTROL"
_QSA_TOPK_MODES = ("radix", "stable", "ordered", "stable_recent", "ordered_recent", "radix_sorted",
                   "stable_desc", "check")
_qsa_topk_mode_cache: list = [0.0, None]
_QSA_STABLE_SUB_ROWS = 256


def _qsa_topk_mode() -> str:
    control = os.environ.get(_QSA_TOPK_CONTROL_ENV)
    if control:
        import time

        now = time.monotonic()
        if _qsa_topk_mode_cache[1] is None or now - _qsa_topk_mode_cache[0] > 0.2:
            try:
                with open(control) as f:
                    mode = f.read().strip()
            except OSError:
                mode = ""
            _qsa_topk_mode_cache[:] = [now, mode or None]
        if _qsa_topk_mode_cache[1]:
            mode = _qsa_topk_mode_cache[1]
            if mode not in _QSA_TOPK_MODES:
                raise ValueError(f"QSA top-k control: unknown mode {mode!r}")
            return mode
    mode = os.environ.get(_QSA_TOPK_ENV, "ordered")
    if mode not in _QSA_TOPK_MODES:
        raise ValueError(f"{_QSA_TOPK_ENV} must be one of {_QSA_TOPK_MODES}, got {mode!r}")
    return mode


_qsa_attn_cache: list = [0.0, None]


def _qsa_attn_precision() -> str:
    """Sparse-attention P.V precision: R9V_QSA_ATTN (default bf16), overridden per call by
    the content of the file named in R9V_QSA_ATTN_CONTROL (research A/B)."""
    control = os.environ.get("R9V_QSA_ATTN_CONTROL")
    if control:
        import time

        now = time.monotonic()
        if _qsa_attn_cache[1] is None or now - _qsa_attn_cache[0] > 0.2:
            try:
                with open(control) as f:
                    value = f.read().strip()
            except OSError:
                value = ""
            _qsa_attn_cache[:] = [now, value or None]
        if _qsa_attn_cache[1]:
            value = _qsa_attn_cache[1]
            if value not in ("bf16", "fp32"):
                raise ValueError(f"QSA attention control: unknown precision {value!r}")
            return value
    value = os.environ.get("R9V_QSA_ATTN", "bf16")
    if value not in ("bf16", "fp32"):
        raise ValueError(f"R9V_QSA_ATTN must be bf16 or fp32, got {value!r}")
    return value


def qsa_stable_block_topk(
    logits: torch.Tensor,
    visible_blocks: torch.Tensor,
    block_topk: int,
    out: torch.Tensor,
    ties_recent: bool = False,
) -> torch.Tensor:
    """Exact per-row top-k of QSA block scores, deterministic.

    Row r ranks columns [0, visible_blocks[r]) by score, ties toward the lower column (the
    higher one with ties_recent), and
    writes the chosen columns in ascending order; rows with at most block_topk visible
    blocks get 0..visible-1. Unused slots are -1. Columns at or past visible_blocks are never
    read as candidates (the scoring kernel leaves columns past its last tile unwritten).
    QSA scores are sums of ReLU'd dot products (finite, >= 0), so their float32 bits order
    like their values once -0.0 is folded into +0.0.
    """
    rows, columns = logits.shape
    col = torch.arange(columns, device=logits.device, dtype=torch.int64)
    tiebreak = col if ties_recent else (columns - 1) - col
    for r0 in range(0, rows, _QSA_STABLE_SUB_ROWS):
        r1 = min(rows, r0 + _QSA_STABLE_SUB_ROWS)
        vis = visible_blocks[r0:r1].to(torch.int64)[:, None]
        valid = col[None, :] < vis
        bits = (logits[r0:r1] + 0.0).view(torch.int32).to(torch.int64).clamp_min_(0)
        key = torch.where(valid, (bits << 32) | tiebreak[None, :], -1)
        top = torch.topk(key, block_topk, dim=1, sorted=False).indices
        top = torch.sort(top, dim=1).values
        out[r0:r1].copy_(torch.where(top < vis, top, -1))
    return out


@triton.jit
def _qsa_ordered_select_kernel(
    logits_ptr,
    visible_ptr,
    kth_ptr,
    out_ptr,
    stride_logits_row,
    stride_out_row,
    num_columns,
    K: tl.constexpr,
    TILE: tl.constexpr,
    TIES_RECENT: tl.constexpr,
) -> None:
    """Write row r's top-k columns in ascending order: every visible column scoring above
    kth[r], then columns scoring exactly kth[r] until k are taken, from the lowest index
    (or, with TIES_RECENT, the highest); -1 in the unused slots. Deterministic: plain
    prefix sums, no atomics."""
    row = tl.program_id(0)
    visible = tl.minimum(tl.load(visible_ptr + row), num_columns)
    kth = tl.load(kth_ptr + row)
    need = tl.minimum(visible, K)
    offsets = tl.arange(0, TILE)
    base = logits_ptr + row.to(tl.int64) * stride_logits_row
    greater = 0
    ties = 0
    for start in tl.range(0, visible, TILE):
        cols = start + offsets
        mask = cols < visible
        score = tl.load(base + cols, mask=mask, other=-float("inf"))
        greater += tl.sum(((score > kth) & mask).to(tl.int32), axis=0)
        ties += tl.sum(((score == kth) & mask).to(tl.int32), axis=0)
    ties_needed = need - greater
    # Tie ranks run from the lowest index; the most recent ties are the last ties_needed.
    first_tie_taken = (ties - ties_needed) * TIES_RECENT
    written = 0
    ties_seen = 0
    out = out_ptr + row.to(tl.int64) * stride_out_row
    for start in tl.range(0, visible, TILE):
        cols = start + offsets
        mask = cols < visible
        score = tl.load(base + cols, mask=mask, other=-float("inf"))
        is_tie = ((score == kth) & mask).to(tl.int32)
        tie_rank = ties_seen + tl.cumsum(is_tie, axis=0) - 1
        take = ((score > kth) & mask) | (
            (is_tie != 0) & (tie_rank >= first_tie_taken) & (tie_rank < first_tie_taken + ties_needed)
        )
        take_i = take.to(tl.int32)
        slot = written + tl.cumsum(take_i, axis=0) - 1
        tl.store(out + slot, cols, mask=take & (slot < K))
        written += tl.sum(take_i, axis=0)
        ties_seen += tl.sum(is_tie, axis=0)
    for start in tl.range(0, K, TILE):
        slots = start + offsets
        tl.store(out + slots, -1, mask=(slots >= written) & (slots < K))


def qsa_ordered_block_topk(
    logits: torch.Tensor,
    visible_blocks: torch.Tensor,
    block_topk: int,
    out: torch.Tensor,
    ties_recent: bool = False,
) -> torch.Tensor:
    """Same result as qsa_stable_block_topk, cheaply: the radix kernel supplies each row's
    k-th best score (a value, identical whichever tied blocks it happened to pick), then
    _qsa_ordered_select_kernel rewrites the selection deterministically in ascending order."""
    rows, columns = logits.shape
    ops.top_k_per_row_decode(
        logits, 1, visible_blocks, out, rows, logits.stride(0), logits.stride(1), block_topk
    )
    picked = torch.gather(logits, 1, out.clamp_min(0).to(torch.int64))
    picked = torch.where(out >= 0, picked, float("inf"))
    kth = picked.min(1).values
    # Rows that see at most k blocks keep all of them: every finite score beats -inf.
    kth = torch.where(visible_blocks > block_topk, kth, float("-inf")).contiguous()
    _qsa_ordered_select_kernel[(rows,)](
        logits,
        visible_blocks,
        kth,
        out,
        logits.stride(0),
        out.stride(0),
        columns,
        K=block_topk,
        TILE=1024,
        TIES_RECENT=int(ties_recent),
        num_warps=4,
    )
    return out


def _sort_selection(blocks: torch.Tensor, keys: torch.Tensor | None = None) -> None:
    """Reorder each row's selected blocks in place, -1 padding kept last: ascending block
    index, or descending keys[row, slot] (ties by index) when keys are given."""
    big = torch.iinfo(torch.int32).max
    if keys is None:
        order = torch.where(blocks >= 0, blocks, big).sort(dim=1).values
        blocks.copy_(torch.where(order == big, -1, order))
        return
    rank = torch.where(blocks >= 0, keys, float("-inf"))
    idx = torch.argsort(rank, dim=1, descending=True, stable=True)
    blocks.copy_(torch.gather(blocks, 1, idx))


_qsa_sel_state: dict = {"count": 0, "weights": {}}


def _qsa_capture_selection(logits, visible_blocks, blocks, query_positions, block_topk) -> None:
    """Research capture (R9V_DET_CAPTURE_DIR with an ENABLE file, GPU 0): per selection call,
    each row's query position, an order-independent hash of its selected block set, and the
    relative gap between its k-th and (k+1)-th best scores (how close the row is to a swap)."""
    root = os.environ.get("R9V_DET_CAPTURE_DIR")
    if (not root or blocks.device.index != 0 or torch.cuda.is_current_stream_capturing()
            or not os.path.exists(os.path.join(root, "ENABLE"))):
        return
    rows, columns = logits.shape
    key = (blocks.device, columns)
    if key not in _qsa_sel_state["weights"]:
        gen = torch.Generator(device="cpu").manual_seed(4321)
        _qsa_sel_state["weights"][key] = torch.randint(
            1, 1 << 40, (columns + 1,), generator=gen, dtype=torch.int64).to(blocks.device)
    weights = _qsa_sel_state["weights"][key]
    sel_hash = weights[torch.where(blocks >= 0, blocks, columns).to(torch.int64)].sum(1)
    col_valid = torch.arange(columns, device=logits.device)[None, :] < visible_blocks[:, None]
    scores = torch.where(col_valid, logits, float("-inf"))
    top = torch.topk(scores, min(block_topk + 1, columns), dim=1).values
    kth, nxt = top[:, block_topk - 1], top[:, block_topk]
    gap = torch.where(visible_blocks > block_topk, (kth - nxt) / kth.abs().clamp_min(1e-30), float("nan"))
    run_file = os.path.join(root, "RUN")
    run = open(run_file).read().strip() if os.path.exists(run_file) else ""
    torch.save({"run": run, "positions": query_positions[:rows].cpu(), "hash": sel_hash.cpu(),
                "gap": gap.cpu()}, os.path.join(root, f"qsel_{_qsa_sel_state['count']:06d}.pt"))
    _qsa_sel_state["count"] += 1


def _qsa_check_topk(
    logits: torch.Tensor, visible_blocks: torch.Tensor, radix: torch.Tensor, block_topk: int
) -> None:
    """Validate radix top-k output against the exact reference; append one JSON line."""
    import json

    rows, columns = logits.shape
    ref = qsa_stable_block_topk(logits, visible_blocks, block_topk, torch.empty_like(radix))
    vis = visible_blocks.to(torch.int64)[:, None]
    need = torch.clamp(vis, max=block_topk)  # slots that must hold a real block
    slot = torch.arange(block_topk, device=radix.device)[None, :]
    used = slot < need
    r = radix.to(torch.int64)
    bad = used & ((r < 0) | (r >= vis))  # includes never-written slots (sentinel -7)
    rs = torch.sort(torch.where(used, r, -1 - slot), dim=1).values
    dup = (rs[:, 1:] == rs[:, :-1]) & (rs[:, 1:] >= 0)
    col_valid = torch.arange(columns, device=logits.device)[None, :] < vis
    scores = torch.where(col_valid, logits, float("-inf"))
    ref_scores = torch.gather(scores, 1, ref.clamp_min(0).to(torch.int64))
    kth = torch.where(ref >= 0, ref_scores, float("inf")).min(1).values  # k-th best score
    got = torch.gather(scores, 1, r.clamp(0, columns - 1))
    non_top = used & ~bad & (got < kth[:, None])
    selective = vis[:, 0] > block_topk
    ties = (col_valid & (scores == kth[:, None])).sum(1)
    above = (col_valid & (scores > kth[:, None])).sum(1)
    in_ref = torch.zeros((rows, columns + 1), dtype=torch.bool, device=radix.device)
    in_ref.scatter_(1, torch.where(ref >= 0, ref.to(torch.int64), columns), True)
    in_radix = torch.gather(in_ref, 1, torch.where(used & ~bad, r, columns))
    missing = (used & ~in_radix).sum(1)  # radix picks not in the reference set
    zeros = (col_valid & (scores == 0)).sum(1)
    positive = (col_valid & (scores > 0)).sum(1)
    # Where the zero-score (tied) blocks sit: the 64 most recent visible blocks vs the rest.
    recent = col_valid & (torch.arange(columns, device=logits.device)[None, :] >= vis - 64)
    zeros_recent = (recent & (scores == 0)).sum(1)
    # Is the query's own (most recent complete) block selected?
    last = (vis[:, 0] - 1).clamp_min(0)
    last_in_radix = (r == last[:, None]).any(1)
    sel = selective.nonzero().flatten()
    stats = {
        "device": radix.device.index,
        "rows": rows,
        "selective_rows": int(selective.sum()),
        "slots_invalid": int(bad.sum()),
        "rows_invalid": int(bad.any(1).sum()),
        "rows_duplicate": int(dup.any(1).sum()),
        "rows_non_top": int(non_top.any(1).sum()),
        "slots_non_top": int(non_top.sum()),
        "rows_set_differs": int((missing > 0).sum()),
        "slots_set_differs": int(missing.sum()),
    }
    if sel.numel():
        t = ties[sel].float()
        stats.update(
            ties_at_kth_median=float(t.median()), ties_at_kth_max=int(t.max()),
            tie_slots_median=float((block_topk - above[sel]).float().median()),
            zero_scores_median=float(zeros[sel].float().median()),
            positive_scores_median=float(positive[sel].float().median()),
            rows_positive_below_k=int((positive[sel] < block_topk).sum()),
            zero_frac_recent64=float(zeros_recent[sel].sum() / (64 * sel.numel())),
            zero_frac_all=float(zeros[sel].sum() / vis[sel].sum()),
            rows_last_block_dropped=int((~last_in_radix[sel]).sum()),
            visible_max=int(vis[sel].max()))
        snap_dir = os.environ.get("R9V_QSA_TOPK_SNAPSHOT_DIR")
        if snap_dir and radix.device.index == 0:
            n = len(os.listdir(snap_dir)) if os.path.isdir(snap_dir) else 0
            if n < 48:
                os.makedirs(snap_dir, exist_ok=True)
                row = int(sel[-1])
                torch.save({"scores": logits[row, : int(vis[row])].cpu(), "radix": radix[row].cpu(),
                            "ref": ref[row].cpu()}, f"{snap_dir}/row_{n:03d}.pt")
    path = os.environ.get("R9V_QSA_TOPK_CHECK_LOG")
    if path:
        with open(f"{path}.dev{radix.device.index}", "a") as f:
            f.write(json.dumps(stats) + "\n")


def qsa_select_paged_tokens(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    token_topk: int,
    compress_ratio: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score, select, and expand QSA indices without host synchronization."""

    rows = q.shape[0]
    output_width = token_topk + compress_ratio - 1
    if out is None:
        out = torch.empty((rows, output_width), dtype=torch.int32, device=q.device)
    if out.shape != (rows, output_width):
        raise ValueError("QSA selection output has an invalid shape")
    if not rows:
        return out

    columns = page_table.shape[1] * k_cache.shape[1]
    block_topk = token_topk // compress_ratio
    rows_per_chunk = max(1, _LOGITS_WORKSPACE_BYTES // max(columns * 4, 1))
    chunk_rows = min(rows, rows_per_chunk)
    blocks_buffer = torch.empty(
        (chunk_rows, block_topk), dtype=torch.int32, device=q.device
    )
    topk_workspace = torch.empty(
        (_TOPK_WORKSPACE_BYTES,), dtype=torch.uint8, device=q.device
    )
    mode = _qsa_topk_mode()
    if mode == "check" and torch.cuda.is_current_stream_capturing():
        mode = "radix"
    for row_start in range(0, rows, rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, rows)
        row_slice = slice(row_start, row_end)
        logits, visible_blocks = qsa_mqa_paged(
            q[row_slice],
            k_cache,
            page_table,
            token_to_req[row_slice],
            query_positions[row_slice],
            sequence_lengths,
            compress_ratio,
        )
        blocks = blocks_buffer[: row_end - row_start]
        use_cooperative_topk = (
            current_platform.is_cuda()
            and blocks.shape[0] <= 32
            and logits.stride(0) % 4 == 0
            and current_platform.has_device_capability(90)
            and not current_platform.is_device_capability_family(120)
        )
        if mode in ("stable", "stable_recent", "stable_desc"):
            qsa_stable_block_topk(logits, visible_blocks, block_topk, blocks, mode == "stable_recent")
            if mode == "stable_desc":
                _sort_selection(blocks, torch.gather(logits, 1, blocks.clamp_min(0).to(torch.int64)))
        elif mode in ("ordered", "ordered_recent"):
            qsa_ordered_block_topk(logits, visible_blocks, block_topk, blocks, mode == "ordered_recent")
        elif use_cooperative_topk:
            torch.ops._C.cooperative_topk(
                logits,
                visible_blocks,
                blocks,
                topk_workspace,
                block_topk,
                columns,
            )
        elif current_platform.is_cuda():
            torch.ops._C.persistent_topk(
                logits,
                visible_blocks,
                blocks,
                topk_workspace,
                block_topk,
                columns,
            )
        else:
            if mode == "check":
                blocks.fill_(-7)  # never-written slots show up as invalid
            ops.top_k_per_row_decode(
                logits,
                1,
                visible_blocks,
                blocks,
                blocks.shape[0],
                logits.stride(0),
                logits.stride(1),
                block_topk,
            )
            if mode == "check":
                _qsa_check_topk(logits, visible_blocks, blocks, block_topk)
            elif mode == "radix_sorted":
                _sort_selection(blocks)
        _qsa_capture_selection(logits, visible_blocks, blocks, query_positions[row_slice], block_topk)
        expand_qsa_block_indices_cuda(
            blocks,
            query_positions[row_slice],
            sequence_lengths,
            token_to_req[row_slice],
            compress_ratio,
            token_topk,
            out[row_slice],
        )
    return out


_qsa_attn_sample_state: dict = {"count": 0}


def _qsa_sample_attention_inputs(q, k_cache, v_cache, logical_indices, block_table, token_to_req):
    """Research capture for offline replays of this kernel on real inputs: with
    R9V_DET_CAPTURE_DIR (+ ENABLE) and R9V_QSA_ATTN_SAMPLE_MIN_POS set, GPU 0 saves the last
    8 query rows of the first calls whose selection reaches past that position, with the
    K/V vectors of every selected token (in selection order)."""
    root = os.environ.get("R9V_DET_CAPTURE_DIR")
    min_pos = int(os.environ.get("R9V_QSA_ATTN_SAMPLE_MIN_POS", "0") or 0)
    if (not root or not min_pos or q.device.index != 0 or _qsa_attn_sample_state["count"] >= 13
            or torch.cuda.is_current_stream_capturing()
            or not os.path.exists(os.path.join(root, "ENABLE")) or q.shape[0] < 8):
        return
    idx = logical_indices[-8:].to(torch.int64)
    if int(idx.max()) < min_pos:
        return
    page = k_cache.shape[1]
    req = token_to_req[-8:].to(torch.int64).clamp_min(0)
    safe = idx.clamp_min(0)
    phys = block_table[req[:, None], (safe // page).clamp_max(block_table.shape[1] - 1)].to(torch.int64)
    off = safe % page
    sample = {"q": q[-8:].cpu(), "indices": logical_indices[-8:].cpu(),
              "k": k_cache[phys, off].cpu(), "v": v_cache[phys, off].cpu()}
    path = os.path.join(root, f"attn_sample_{_qsa_attn_sample_state['count']:02d}.pt")
    torch.save(sample, path)
    _qsa_attn_sample_state["count"] += 1


def qsa_sparse_paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    out: torch.Tensor | None = None,
    fp32_pv: bool | None = None,
) -> torch.Tensor:
    """Run sparse GQA directly over paged BF16 K/V caches.

    R9V: fp32_pv (default from R9V_QSA_ATTN / the R9V_QSA_ATTN_CONTROL file, "bf16" or
    "fp32") keeps the softmax probabilities in fp32 for the P.V product. A float32 `out`
    is accepted for research replays."""
    if fp32_pv is None:
        fp32_pv = _qsa_attn_precision() == "fp32"
    _qsa_sample_attention_inputs(q, k_cache, v_cache, logical_indices, block_table, token_to_req)

    if not q.is_cuda or not HAS_TRITON:
        raise RuntimeError("paged QSA sparse attention requires a GPU and Triton")
    if q.ndim != 3 or k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
        raise ValueError("QSA sparse attention received invalid Q/K/V shapes")
    if logical_indices.ndim != 2 or logical_indices.shape[0] != q.shape[0]:
        raise ValueError("QSA indices must have one row per query")
    if token_to_req.shape != (q.shape[0],) or block_table.ndim != 2:
        raise ValueError("QSA sparse attention metadata has invalid shapes")
    if not all(k_cache.shape[:3]) or not all(block_table.shape):
        raise ValueError("QSA sparse attention cache and block table must be nonempty")
    if logical_indices.shape[1] <= 0:
        raise ValueError("QSA sparse attention requires a positive selection width")
    if q.shape[2] != k_cache.shape[3] or q.shape[1] % k_cache.shape[2]:
        raise ValueError("QSA sparse attention requires valid grouped-query heads")
    head_dim = q.shape[2]
    assert head_dim >= 16 and (head_dim & (head_dim - 1)) == 0
    assert q.dtype == k_cache.dtype == v_cache.dtype == torch.bfloat16
    assert logical_indices.dtype == block_table.dtype == torch.int32
    assert token_to_req.dtype == torch.int32
    assert q.device == k_cache.device == v_cache.device
    assert q.device == logical_indices.device == block_table.device
    assert q.device == token_to_req.device
    assert q.stride(2) == k_cache.stride(3) == v_cache.stride(3) == 1
    assert logical_indices.stride(1) == block_table.stride(1) == 1
    assert token_to_req.stride(0) == 1
    if out is None:
        out = torch.empty_like(q)
    if out.shape != q.shape:
        raise ValueError("QSA sparse output must match its query")
    assert out.dtype in (q.dtype, torch.float32) and out.device == q.device
    assert out.stride(2) == 1
    if not q.shape[0]:
        return out

    group_size = q.shape[1] // k_cache.shape[2]
    block_m = triton.next_power_of_2(group_size)
    base_programs = q.shape[0] * k_cache.shape[2]
    small_profile_limit = 8 if block_m <= 8 else 4

    # Tuned on GB300 for the Qwen-Air TP1, TP2, and TP4 attention shapes.
    # Narrow tiles favor decode; wide tiles improve throughput for prefill.
    if base_programs <= small_profile_limit:
        block_n, target_splits, partial_warps = 16, 64, 4
    elif base_programs < 32:
        block_n, target_splits, partial_warps = 16, 32, 4
    elif base_programs <= 256:
        block_n, target_splits, partial_warps = 64, 8, 2
    elif base_programs <= 512:
        block_n, target_splits, partial_warps = 64, 4, 2
    else:
        block_n, target_splits, partial_warps = 64, 1, 2
    # gfx942 and gfx950 have a 64 KiB LDS limit. One software-pipelining
    # stage keeps the wide TP4 tile within that shared-memory budget.
    partial_stages = 1 if current_platform.is_rocm() else 2
    if fp32_pv:
        block_n = 16  # fp32 V tiles: keep the P.V operands within the LDS budget

    num_tiles = triton.cdiv(logical_indices.shape[1], block_n)
    # Avoid empty splits when the selection width is smaller than the profile.
    max_useful_splits = 1 << (num_tiles.bit_length() - 1)
    num_splits = min(max_useful_splits, target_splits)

    # Split=1 writes output directly and compiles out all workspace accesses.
    if num_splits == 1:
        partial_output = out
        partial_lse = out
    else:
        # FP32 partials preserve accuracy when merging independently normalized
        # splits.
        partial_output = torch.empty(
            (num_splits, *q.shape), dtype=torch.float32, device=q.device
        )
        partial_lse = torch.empty(
            (num_splits, q.shape[0], q.shape[1]),
            dtype=torch.float32,
            device=q.device,
        )

    partial_grid = (q.shape[0], k_cache.shape[2], num_splits)
    _qsa_sparse_paged_gqa_splitk_kernel[partial_grid](
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_req,
        partial_output,
        partial_lse,
        out,
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        logical_indices.stride(0),
        block_table.stride(0),
        out.stride(0),
        out.stride(1),
        q.shape[0],
        k_cache.shape[0],
        block_table.shape[0],
        TOPK=logical_indices.shape[1],
        PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=block_table.shape[1],
        GROUP_SIZE=group_size,
        HEAD_DIM=q.shape[2],
        NUM_QUERY_HEADS=q.shape[1],
        NUM_SPLITS=num_splits,
        NUM_TILES=num_tiles,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        FP32_PV=bool(fp32_pv),
        num_warps=partial_warps,
        num_stages=partial_stages,
    )
    if num_splits == 1:
        return out

    _qsa_merge_splitk_kernel[(q.shape[0], q.shape[1])](
        partial_output,
        partial_lse,
        out,
        out.stride(0),
        out.stride(1),
        q.shape[0],
        HEAD_DIM=q.shape[2],
        NUM_QUERY_HEADS=q.shape[1],
        NUM_SPLITS=num_splits,
        BLOCK_SPLITS=triton.next_power_of_2(num_splits),
        num_warps=2,
        num_stages=1,
    )
    return out


def qsa_store_cache_rows(
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    rows: torch.Tensor,
) -> None:
    """Store fixed-width rows in a QSA cache without boolean indexing."""

    if not cache.is_cuda or not HAS_TRITON:
        raise RuntimeError("QSA cache stores require a GPU and Triton")
    if cache.ndim != 4 or cache.shape[2] != 1:
        raise ValueError("QSA cache must be [pages, page_size, 1, width]")
    if not all(cache.shape):
        raise ValueError("QSA cache dimensions must be nonzero")
    if rows.ndim == 3:
        if rows.shape[1] != 1:
            raise ValueError("QSA cache rows must have one head")
        rows = rows[:, 0]
    if rows.shape != (slot_mapping.numel(), cache.shape[3]):
        raise ValueError("QSA cache rows and slots have incompatible shapes")
    if not rows.shape[0]:
        return
    _store_qsa_rows_kernel[(rows.shape[0],)](
        cache,
        slot_mapping,
        rows,
        cache.stride(0),
        cache.stride(1),
        cache.stride(3),
        rows.stride(0),
        rows.stride(1),
        rows.shape[0],
        cache.shape[0],
        PAGE_SIZE=cache.shape[1],
        WIDTH=cache.shape[3],
        BLOCK_D=triton.next_power_of_2(cache.shape[3]),
        num_warps=4,
    )


def qsa_compress_groups_with_ratio(
    raw_keys: torch.Tensor,  # this step's raw key rows [rows, 1, head_size]
    raw_positions: torch.Tensor,  # this step's positions [rows, 1, 3] int64
    compressor_state_cache: torch.Tensor,
    compressor_state_block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_start_loc: torch.Tensor,
    logical_positions: torch.Tensor,
    compressed_slots: torch.Tensor,
    compress_ratio: int,
    rope_cache: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool completed groups from the compressor-state ring and raw token rows."""

    if not raw_keys.is_cuda or not HAS_TRITON:
        raise RuntimeError("QSA compression requires a GPU and Triton")
    rows = token_to_req.numel()
    if compress_ratio <= 0:
        raise ValueError("QSA compression ratio must be positive")
    if raw_keys.ndim != 3 or raw_keys.shape[:2] != (rows, 1):
        raise ValueError("QSA raw keys must be [rows, 1, head_size]")
    if raw_positions.shape != (rows, 1, 3) or raw_positions.dtype != torch.int64:
        raise ValueError("QSA raw positions must be [rows, 1, 3] int64")
    if logical_positions.shape != (rows,) or compressed_slots.shape != (rows,):
        raise ValueError("QSA compression metadata must match token rows")
    if compressor_state_cache.ndim != 4 or compressor_state_cache.shape[2] != 1:
        raise ValueError("QSA compressor-state cache has an invalid shape")
    if (
        # The ring is wider than one group so speculative rows cannot alias
        # onto the committed keys of the group still being collected.
        compressor_state_cache.shape[1] < compress_ratio
        or compressor_state_cache.shape[3] != raw_keys.shape[2]
        or compressor_state_cache.dtype != raw_keys.dtype
    ):
        raise ValueError(
            "QSA compressor-state cache does not match the compression layout"
        )
    if (
        compressor_state_block_table.ndim != 2
        or compressor_state_block_table.shape[1] < 1
    ):
        raise ValueError(
            "QSA compressor-state block table must contain one block per request"
        )
    if query_start_loc.ndim != 1 or query_start_loc.shape[0] < 2:
        raise ValueError("QSA query starts must contain a terminal offset")
    num_requests = query_start_loc.shape[0] - 1
    if compressor_state_block_table.shape[0] < num_requests:
        raise ValueError("QSA compressor-state block table has too few request rows")
    if rope_cache is not None and (
        rope_cache.ndim != 4
        or rope_cache.shape[:3] != compressor_state_cache.shape[:3]
        or rope_cache.shape[3] != 3
        or rope_cache.dtype != torch.int64
    ):
        raise ValueError("QSA packed position view has an invalid shape or dtype")
    if rows and (
        not all(compressor_state_cache.shape)
        or not all(compressor_state_block_table.shape)
    ):
        raise ValueError("QSA compressor-state cache and block table must be nonempty")
    pooled = torch.empty(
        (rows, 1, raw_keys.shape[2]),
        dtype=raw_keys.dtype,
        device=raw_keys.device,
    )
    first_positions = torch.empty((rows, 3), dtype=torch.int64, device=raw_keys.device)
    if not rows:
        return pooled, first_positions
    if rope_cache is None:
        rope_cache = compressor_state_cache
        load_rope_positions = False
    else:
        load_rope_positions = True
    _compress_qsa_groups_kernel[(rows,)](
        raw_keys,
        raw_positions,
        compressor_state_cache,
        rope_cache,
        compressor_state_block_table,
        token_to_req,
        query_start_loc,
        logical_positions,
        compressed_slots,
        pooled,
        first_positions,
        raw_keys.stride(0),
        raw_keys.stride(2),
        raw_positions.stride(0),
        raw_positions.stride(2),
        compressor_state_cache.stride(0),
        compressor_state_cache.stride(1),
        compressor_state_cache.stride(3),
        rope_cache.stride(0),
        rope_cache.stride(1),
        rope_cache.stride(3),
        compressor_state_block_table.stride(0),
        pooled.stride(0),
        pooled.stride(2),
        first_positions.stride(0),
        first_positions.stride(1),
        rows,
        compressor_state_cache.shape[0],
        num_requests,
        COMPRESSOR_STATE_SIZE=compressor_state_cache.shape[1],
        COMPRESS_RATIO=compress_ratio,
        HEAD_DIM=raw_keys.shape[2],
        LOAD_ROPE_POSITIONS=load_rope_positions,
        BLOCK_D=triton.next_power_of_2(raw_keys.shape[2]),
        num_warps=4,
    )
    return pooled, first_positions


__all__ = [
    "expand_qsa_block_indices_cuda",
    "qsa_compress_groups_with_ratio",
    "qsa_mqa_paged",
    "qsa_select_paged_tokens",
    "qsa_sparse_paged_attention",
    "qsa_store_cache_rows",
]
