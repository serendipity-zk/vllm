# SPDX-License-Identifier: Apache-2.0
"""MoESim-only routed expert summary tracing for vLLM profiling runs."""

from __future__ import annotations

import fcntl
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch


def is_enabled() -> bool:
    return bool(os.environ.get("VLLM_MOESIM_ROUTING_TRACE_PATH"))


def is_token_trace_enabled() -> bool:
    return bool(os.environ.get("VLLM_MOESIM_TOKEN_TRACE_PATH"))


def _parse_iterations(raw: str) -> set[int] | None:
    raw = raw.strip()
    if not raw:
        return None

    selected: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            selected.update(range(start, end + 1))
        else:
            selected.add(int(part))
    return selected


def _selected_iterations() -> set[int] | None:
    return _parse_iterations(os.environ.get("VLLM_MOESIM_ROUTING_TRACE_ITERS", ""))


def _selected_token_iterations() -> set[int] | None:
    token_iters = os.environ.get("VLLM_MOESIM_TOKEN_TRACE_ITERS", "")
    if token_iters.strip():
        return _parse_iterations(token_iters)
    return _selected_iterations()


def should_trace_iteration(iteration_index: int | None) -> bool:
    if not is_enabled():
        return False
    selected = _selected_iterations()
    if selected is None:
        return True
    return iteration_index in selected


def should_trace_token_iteration(iteration_index: int | None) -> bool:
    if not is_token_trace_enabled():
        return False
    selected = _selected_token_iterations()
    if selected is None:
        return True
    return iteration_index in selected


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            handle.write(line)
            handle.flush()
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _percentile(values: list[int], pct: float) -> int:
    if not values:
        return 0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(round((len(values) - 1) * pct))))
    return values[idx]


def _tensor_int_list(tensor: torch.Tensor) -> list[int]:
    return [int(v) for v in tensor.detach().cpu().tolist()]


def _seq_int_list(values: Any) -> list[int]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [int(v) for v in values]


def dump_token_inputs(
    *,
    iteration_index: int | None,
    input_ids: torch.Tensor | None,
    req_ids: list[str],
    num_scheduled_tokens: Any,
    num_tokens: int,
    positions: Any | None = None,
) -> None:
    """Dump scheduled input token IDs for MoESim alignment-only reruns."""
    if not should_trace_token_iteration(iteration_index):
        return
    if input_ids is None or num_tokens <= 0:
        return

    trace_path_raw = os.environ.get("VLLM_MOESIM_TOKEN_TRACE_PATH")
    if not trace_path_raw:
        return

    num_tokens = int(num_tokens)
    token_ids = _tensor_int_list(input_ids[:num_tokens])
    scheduled = _seq_int_list(num_scheduled_tokens)
    positions_list: list[int] | None = None
    if positions is not None:
        positions_list = _seq_int_list(positions[:num_tokens])

    requests: list[dict[str, Any]] = []
    offset = 0
    for req_id, count in zip(req_ids, scheduled):
        end = offset + count
        request_row: dict[str, Any] = {
            "req_id": req_id,
            "start": offset,
            "end": end,
            "num_scheduled_tokens": count,
            "token_ids": token_ids[offset:end],
        }
        if positions_list is not None:
            request_row["positions"] = positions_list[offset:end]
        requests.append(request_row)
        offset = end

    row: dict[str, Any] = {
        "timestamp_unix": time.time(),
        "pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "local_cuda_device": (
            torch.cuda.current_device() if torch.cuda.is_available() else None
        ),
        "iteration": iteration_index,
        "tokens": num_tokens,
        "num_reqs": len(req_ids),
        "req_ids": req_ids,
        "num_scheduled_tokens": scheduled,
        "token_ids_flat": token_ids,
        "requests": requests,
    }
    if positions_list is not None:
        row["positions_flat"] = positions_list

    _append_jsonl(Path(trace_path_raw), row)


def _iter_moe_layers(static_forward_context: dict[str, Any]):
    for layer_name, layer in static_forward_context.items():
        if not hasattr(layer, "global_num_experts"):
            continue
        if not hasattr(layer, "local_num_experts"):
            continue
        layer_id = getattr(layer, "layer_id", None)
        if layer_id is None:
            continue
        yield int(layer_id), layer_name, layer


def dump_routing_summary(
    *,
    capturer: Any,
    static_forward_context: dict[str, Any],
    iteration_index: int | None,
    num_tokens: int,
) -> None:
    """Dump per-layer routed expert histograms from the capturer device buffer."""
    if not should_trace_iteration(iteration_index):
        return
    if capturer is None or getattr(capturer, "_device_buffer", None) is None:
        return

    trace_path_raw = os.environ.get("VLLM_MOESIM_ROUTING_TRACE_PATH")
    if not trace_path_raw:
        return
    trace_path = Path(trace_path_raw)

    device_buffer = capturer._device_buffer
    if num_tokens <= 0:
        return
    num_tokens = min(int(num_tokens), int(device_buffer.shape[0]))

    block_m = int(os.environ.get("VLLM_MOESIM_ROUTING_TRACE_BLOCK_M", "64"))
    include_global_counts = (
        os.environ.get("VLLM_MOESIM_ROUTING_TRACE_GLOBAL_COUNTS", "1") != "0"
    )
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    local_cuda_device = torch.cuda.current_device() if torch.cuda.is_available() else None

    for layer_id, layer_name, layer in _iter_moe_layers(static_forward_context):
        if layer_id >= int(device_buffer.shape[1]):
            continue
        topk_ids = device_buffer[:num_tokens, layer_id, :]
        if topk_ids.numel() == 0:
            continue

        topk = int(topk_ids.shape[1])
        global_num_experts = int(layer.global_num_experts)
        flat = topk_ids.reshape(-1)
        valid_mask = flat >= 0
        valid_flat = flat[valid_mask].to(torch.int64)
        global_counts_t = torch.bincount(
            valid_flat, minlength=global_num_experts
        )[:global_num_experts]
        global_counts = _tensor_int_list(global_counts_t)

        local_counts: list[int]
        local_global_experts: list[int]
        expert_map = layer.expert_map
        if expert_map is not None:
            expert_map_t = expert_map.to(device=flat.device)
            clamped_flat = flat.clamp(min=0).to(torch.int64)
            local_ids_full = expert_map_t[clamped_flat]
            local_mask = valid_mask & (local_ids_full >= 0)
            local_ids = local_ids_full[local_mask].to(torch.int64)
            local_counts_t = torch.bincount(
                local_ids, minlength=int(layer.local_num_experts)
            )[: int(layer.local_num_experts)]
            local_counts = _tensor_int_list(local_counts_t)
            local_global_experts = _tensor_int_list(torch.where(expert_map_t >= 0)[0])
        else:
            local_counts = global_counts
            local_global_experts = list(range(global_num_experts))

        local_assignments = int(sum(local_counts))
        total_assignments = int(valid_flat.numel())
        nonzero_local_counts = [count for count in local_counts if count > 0]
        local_padded = sum(
            int(math.ceil(count / block_m) * block_m)
            for count in nonzero_local_counts
        )
        sorted_token_ids_len = total_assignments + global_num_experts * (block_m - 1)
        launch_m_blocks = int(math.ceil(sorted_token_ids_len / block_m))
        effective_m_blocks = int(math.ceil(local_padded / block_m)) if local_padded else 0

        row: dict[str, Any] = {
            "timestamp_unix": time.time(),
            "pid": os.getpid(),
            "cuda_visible_devices": cuda_visible_devices,
            "local_cuda_device": local_cuda_device,
            "iteration": iteration_index,
            "layer_id": layer_id,
            "layer_name": layer_name,
            "tokens": num_tokens,
            "top_k": topk,
            "total_assignments": total_assignments,
            "global_num_experts": global_num_experts,
            "ep_size": int(getattr(layer, "ep_size", 1)),
            "ep_rank": int(getattr(layer, "ep_rank", 0)),
            "tp_size": int(getattr(layer, "tp_size", 1)),
            "tp_rank": int(getattr(layer, "tp_rank", 0)),
            "local_num_experts": int(layer.local_num_experts),
            "local_global_experts": local_global_experts,
            "local_counts": local_counts,
            "local_assignments": local_assignments,
            "local_count_min": min(local_counts) if local_counts else 0,
            "local_count_p50": _percentile(local_counts, 0.5),
            "local_count_p90": _percentile(local_counts, 0.9),
            "local_count_max": max(local_counts) if local_counts else 0,
            "local_nonzero_experts": len(nonzero_local_counts),
            "block_m_assumed": block_m,
            "local_padded": local_padded,
            "sorted_token_ids_len": sorted_token_ids_len,
            "launch_m_blocks": launch_m_blocks,
            "effective_m_blocks": effective_m_blocks,
            "m_block_overlaunch": (
                launch_m_blocks / effective_m_blocks if effective_m_blocks else None
            ),
        }
        if include_global_counts:
            row["global_counts"] = global_counts

        _append_jsonl(trace_path, row)
