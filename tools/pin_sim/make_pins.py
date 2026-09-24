#!/usr/bin/env python3
"""Write the full mutable cache's permanent expert pins from a saved routing trace.

Per layer, rank 1 pins its most-routed experts over the trace's train split (ties broken by
lower expert id); rank 0 pins none. The output records the trace hash and split so the file
can be regenerated exactly.
Usage: make_pins.py ROUTES_NPZ OUT_JSON
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

PINNED = {"0": 0, "1": 400}


def top_experts(routes, count):
    """Most-routed expert ids, most first; ties go to the lower id."""
    counts = np.bincount(routes.ravel(), minlength=512)
    return [int(e) for e in np.argsort(-counts, kind="stable")[:count]]


def main():
    routes_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    data = np.load(routes_path)
    train = data["splits"] == "train"
    routes = data["routes"][train]
    pins = {rank: [top_experts(routes[:, layer], count) for layer in range(routes.shape[1])]
            for rank, count in PINNED.items()}
    provenance = {
        "trace_file": routes_path.name,
        "trace_sha256": hashlib.sha256(routes_path.read_bytes()).hexdigest(),
        "split": "train",
        "train_events": int(train.sum()),
        "train_prompt_indices": sorted({int(p) for p in data["prompt_indices"][train]}),
        "selection": "per layer, top-N experts by route count over train events; ties to lower id",
        "generator": "tools/pin_sim/make_pins.py",
    }
    out_path.write_text(json.dumps({"version": 1, "pinned_slots": PINNED, "provenance": provenance,
                                    "pinned_experts_by_layer": pins}, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
