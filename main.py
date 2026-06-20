#!/usr/bin/env python3
"""TrafficGuard AI CLI entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pipeline.processor import TrafficGuardPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="TrafficGuard AI - Traffic Violation Detection")
    sub = parser.add_subparsers(dest="command", required=True)

    p_img = sub.add_parser("process", help="Process a single traffic image")
    p_img.add_argument("image", type=str, help="Path to image file")
    p_img.add_argument("--config", type=str, default=None)
    p_img.add_argument("--no-save", action="store_true")

    p_batch = sub.add_parser("batch", help="Process all images in a directory")
    p_batch.add_argument("directory", type=str)
    p_batch.add_argument("--config", type=str, default=None)

    p_report = sub.add_parser("report", help="Generate analytics report")
    p_report.add_argument("--days", type=int, default=30)
    p_report.add_argument("--config", type=str, default=None)

    p_search = sub.add_parser("search", help="Search violation records")
    p_search.add_argument("--type", type=str, default=None)
    p_search.add_argument("--plate", type=str, default=None)
    p_search.add_argument("--config", type=str, default=None)

    args = parser.parse_args()
    pipeline = TrafficGuardPipeline(config_path=args.config)

    if args.command == "process":
        result = pipeline.process_image(args.image, save_to_db=not args.no_save)
        print(json.dumps(result.to_dict(), indent=2))

    elif args.command == "batch":
        results = pipeline.process_batch(args.directory)
        print(f"Processed {len(results)} images.")
        total_violations = sum(len(r.violations) for r in results)
        print(f"Total violations detected: {total_violations}")

    elif args.command == "report":
        analytics = pipeline.get_analytics(days=args.days)
        csv_path = pipeline.reporter.export_csv()
        print(json.dumps(analytics, indent=2))
        print(f"\nCSV exported to: {csv_path}")

    elif args.command == "search":
        records = pipeline.db.search(violation_type=args.type, plate_number=args.plate)
        print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
