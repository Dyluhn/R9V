#!/usr/bin/env python3
"""Replay expert routing through a full_mutable_device build; print per-event state digests and timing.

Run once per build (separate processes, same module name), then diff the digest files.
Usage: planner_equivalence.py SO_PATH ROUTES_NPZ PLACEMENT_JSON OUT_JSON
"""
import hashlib
import importlib.util
import json
import sys
import time

import numpy as np
import torch

HOT_SLOTS = {0: 62, 1: 427}
CACHE_SLOTS = {0: 155, 1: 0}
CAPACITY = {0: 217, 1: 427}
ROWS, W13_BYTES, W2_BYTES = 2, 32, 16  # tiny fake expert payloads; the planner only moves pointers/bytes


def load(so_path):
    spec = importlib.util.spec_from_file_location("qwen38_full_mutable_device", so_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Layer:
    def __init__(self, ext, rank, hot_ids):
        dev = torch.device("cuda")
        self.ext, self.rank = ext, rank
        ids = torch.arange(512, dtype=torch.uint8, device=dev)
        self.cold13 = ids.view(512, 1, 1).expand(512, ROWS, W13_BYTES).contiguous()
        self.cold2 = (255 - ids).view(512, 1, 1).expand(512, ROWS, W2_BYTES).contiguous()
        self.hot13 = torch.zeros(HOT_SLOTS[rank], ROWS, W13_BYTES, dtype=torch.uint8, device=dev)
        self.hot2 = torch.zeros(HOT_SLOTS[rank], ROWS, W2_BYTES, dtype=torch.uint8, device=dev)
        self.cache13 = self.cache2 = self.cache_map = None
        if CACHE_SLOTS[rank]:
            self.cache13 = torch.zeros(CACHE_SLOTS[rank], ROWS, W13_BYTES, dtype=torch.uint8, device=dev)
            self.cache2 = torch.zeros(CACHE_SLOTS[rank], ROWS, W2_BYTES, dtype=torch.uint8, device=dev)
            self.cache_map = torch.full((512,), -1, dtype=torch.int32, device=dev)
        self.hot_map = torch.full((512,), -1, dtype=torch.int32, device=dev)
        self.cold_map = torch.full((512,), -1, dtype=torch.int32, device=dev)
        self.arena = torch.zeros(32768, dtype=torch.uint8, device=dev)
        hot = torch.tensor(hot_ids[:HOT_SLOTS[rank]], dtype=torch.int32, device=dev)
        ext.full_mutable_device_init(self.arena, hot, self.hot_map, self.cold_map, self.cache_map,
                                     rank, CAPACITY[rank])
        # Mirror production warm start: hot slots hold their experts' bytes.
        self.hot13.copy_(self.cold13[hot.long()])
        self.hot2.copy_(self.cold2[hot.long()])

    def step(self, expert_ids):
        self.ext.full_mutable_device_step(
            self.arena, expert_ids, self.cold13, self.cold2, self.hot13, self.hot2,
            self.hot_map, self.cold_map, self.cache13, self.cache2, self.cache_map,
            self.rank, CAPACITY[self.rank], 64)

    def digest(self):
        parts = [self.arena, self.hot_map, self.hot13, self.hot2]
        if self.cache_map is not None:
            parts += [self.cache_map, self.cache13, self.cache2]
        h = hashlib.sha256()
        for p in parts:
            h.update(p.cpu().numpy().tobytes())
        return h.hexdigest()[:16]


def event_streams(routes, rng):
    """(name, list of int32 route tensors) per stream: real 5-token, real 1-token, random with repeats."""
    streams = {}
    for layer in (0, 17, 47):
        streams[f"trace-L{layer}-5tok"] = [routes[e, layer].reshape(5, 10) for e in range(routes.shape[0])]
        streams[f"trace-L{layer}-1tok"] = [routes[e, layer, :1].reshape(1, 10) for e in range(routes.shape[0])]
    # Random events: small id pool forces duplicates and hits; wide pool forces misses/evictions.
    streams["random-narrow"] = [rng.integers(0, 40, size=(5, 10)) for _ in range(400)]
    streams["random-wide"] = [rng.integers(0, 512, size=(5, 10)) for _ in range(400)]
    return streams


def main():
    so_path, routes_path, placement_path, out_path = sys.argv[1:5]
    ext = load(so_path)
    routes = np.load(routes_path)["routes"].astype(np.int32)
    placement = json.loads(open(placement_path).read())
    rng = np.random.default_rng(1234)
    result = {"so": hashlib.sha256(open(so_path, "rb").read()).hexdigest(), "streams": {}, "timing_us": {}}
    for name, events in event_streams(routes, rng).items():
        layer = int(name.split("-L")[1].split("-")[0]) if name.startswith("trace") else 0
        for rank in (0, 1):
            hot_ids = placement["ranks"][str(rank)]["hot_experts_by_layer"][layer]
            state = Layer(ext, rank, hot_ids)
            dev_events = [torch.tensor(e, dtype=torch.int32, device="cuda") for e in events]
            digests = []
            for ids in dev_events:
                state.step(ids)
                digests.append(state.digest())
            result["streams"][f"{name}-rank{rank}"] = digests
            if name == "trace-L17-5tok":  # time steady-state steps (planner + gather + publish)
                timed = Layer(ext, rank, hot_ids)
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(3):
                    for ids in dev_events:
                        timed.step(ids)
                torch.cuda.synchronize()
                result["timing_us"][f"rank{rank}"] = (time.perf_counter() - start) / (3 * len(dev_events)) * 1e6
    open(out_path, "w").write(json.dumps(result))
    print("streams", len(result["streams"]), "timing_us", result["timing_us"])


if __name__ == "__main__":
    main()
