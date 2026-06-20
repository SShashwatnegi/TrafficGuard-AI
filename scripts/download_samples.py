#!/usr/bin/env python3
"""Download sample traffic images for testing TrafficGuard AI."""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

SAMPLES = [
    (
        "traffic_intersection.jpg",
        "https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=800&q=80",
    ),
    (
        "highway_traffic.jpg",
        "https://images.unsplash.com/photo-1519003722824-194d4455a60c?w=800&q=80",
    ),
    (
        "city_street.jpg",
        "https://images.unsplash.com/photo-1502877338535-766e1452684a?w=800&q=80",
    ),
]


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "data" / "sample"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, url in SAMPLES:
        dest = out_dir / name
        if dest.exists():
            print(f"Skip (exists): {dest}")
            continue
        print(f"Downloading {name}...")
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"  Saved to {dest}")
        except Exception as e:
            print(f"  Failed: {e}", file=sys.stderr)

    print(f"\nSample images ready in: {out_dir}")
    print("Run: python main.py batch data/sample")


if __name__ == "__main__":
    main()
