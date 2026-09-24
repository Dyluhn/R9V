#!/usr/bin/env python3
"""Size permanent expert pinning: replay saved routing through the planner at several pin counts.

Usage: run.py PIN_SIM_BINARY ROUTES_NPZ PLACEMENT_JSON OUT_JSON
Build the binary from the runtime's cache sources (CPU only):
  g++ -O2 -std=c++17 -I runtimes/qwen38-flash-next-gfx1201-mtp4-v3/sources/cache \
      tools/pin_sim/pin_sim.cpp -o pin_sim
ROUTES_NPZ is a saved decode routing trace (routes [events, 48 layers, 50 ids], splits,
prompt_indices); full_mutable_pins.json records the SHA-256 of the one the pins came from.
Pins are chosen two ways: the placement's priority order, or the most-routed experts in the
train split. Results are scored on the holdout split (state carried over from train).
"""
import json
import subprocess
import sys
import tempfile

import numpy as np

PINS = {0: [0, 20, 40, 62], 1: [0, 200, 300, 350, 400, 427]}
HOT = {0: 62, 1: 427}


def warm_list(priority, pins, nhot):
    rest = [e for e in priority if e not in set(pins)]
    return (list(pins) + rest)[:nhot]


def main():
    binary, routes_path, placement_path, out_path = sys.argv[1:5]
    data = np.load(routes_path)
    routes = data["routes"].astype(np.int32).reshape(len(data["routes"]), 48, 50)
    train = int((data["splits"] == "train").sum())
    assert (data["splits"][:train] == "train").all()
    placement = json.load(open(placement_path))
    layer_bytes = placement["expert_memory"]["packed_bytes_per_expert_by_layer_by_rank"]
    lines, keys = [], []
    for rank in (0, 1):
        for layer in range(48):
            priority = placement["ranks"][str(rank)]["hot_experts_by_layer"][layer]
            counts = np.bincount(routes[:train, layer].ravel(), minlength=512)
            by_use = [int(e) for e in np.argsort(-counts, kind="stable")]
            for pinned in PINS[rank]:
                for choice, order in (("priority", priority), ("train_top", by_use)):
                    warm = warm_list(priority, order[:pinned], HOT[rank])
                    lines.append(f"{rank} {layer} {pinned} {len(warm)} " + " ".join(map(str, warm)))
                    keys.append(choice)
    with tempfile.NamedTemporaryFile(suffix=".bin") as blob:
        routes.tofile(blob.name)
        out = subprocess.run([binary, blob.name, str(train)], input="\n".join(lines), text=True,
                             capture_output=True, check=True, timeout=3600).stdout.split("\n")
    result = {}
    for index, key in enumerate(keys):
        for row in out[2 * index:2 * index + 2]:
            rank, layer, pinned, split, admitted, uncached, token_miss = row.split()
            per = layer_bytes[rank][int(layer)]
            cell = result.setdefault(f"rank{rank}/{key}/pin{pinned}/{split}", dict.fromkeys(
                ("admitted_bytes", "uncached_bytes", "unique_misses", "token_misses"), 0))
            cell["admitted_bytes"] += int(admitted) * per
            cell["uncached_bytes"] += int(uncached) * per
            cell["unique_misses"] += int(admitted) + int(uncached)
            cell["token_misses"] += int(token_miss)
    saved = {f"rank{r}/pin{p}": p * sum(layer_bytes[str(r)]) for r in (0, 1) for p in PINS[r]}
    json.dump({"events": {"train": train, "holdout": len(routes) - train}, "cells": result,
               "host_bytes_saved": saved}, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main()
