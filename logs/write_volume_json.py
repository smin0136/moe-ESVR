"""Backfill volume.json into finished model folders (train_multivar.py writes it for new runs).

Usage: python logs/write_volume_json.py [root ...]   (default: output/multivar)
The shape (X, Y, Z) is read from each folder's scan_conversion_log.txt.
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.train_utils import write_volume_json

roots = sys.argv[1:] or ["output/multivar"]
written = 0
for root in roots:
    for dirpath, dirnames, filenames in os.walk(root):
        if "scan_conversion_log.txt" not in filenames or "routing.json" not in filenames:
            continue
        with open(os.path.join(dirpath, "scan_conversion_log.txt")) as f:
            match = re.search(r"volume shape\s*:\s*\((\d+),\s*(\d+),\s*(\d+)\)", f.read())
        if match is None:
            print(f"skip (no shape): {dirpath}")
            continue
        write_volume_json(dirpath, tuple(int(n) for n in match.groups()))
        written += 1
print(f"wrote volume.json in {written} folders")
