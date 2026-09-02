#!/usr/bin/env python3
"""08: Classify completed BugPicker plate specimens from their saved multi-view images."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from taxonomy_classifier import TaxonomyClassifier

IMAGE_KINDS = ("scan", "well", "bottom", "hires")
INVALID_IMAGE_CODES = {"", "nan", "none", "null", "<blank>", "blank"}

AI_COLUMNS = [
    "AI Order",
    "AI Order Confidence",
    "AI Order Top 3",
    "AI Species",
    "AI Species Confidence",
    "AI Species Top 3",
    "AI Taxonomy Model",
    "AI Taxonomy Fusion",
    "AI Taxonomy Views",
    "AI Taxonomy Status",
    "AI Taxonomy Error",
    "AI Taxonomy Timestamp",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_plate(value: str) -> str:
    plate = "".join(ch for ch in value.strip().upper() if ch.isalnum() or ch in "_-")
    if not plate:
        raise ValueError("Plate number must not be blank")
    return plate


def normalize_image_code(value: Any) -> str:
    """Return a usable Image Code, or an empty string for placeholder/blank cells."""
    if value is None:
        return ""
    image_code = str(value).strip()
    if image_code.lower() in INVALID_IMAGE_CODES:
        return ""
    return image_code


def find_image_paths(image_dir: Path, image_code: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for kind in IMAGE_KINDS:
        path = image_dir / f"{image_code}_{kind}.png"
        if path.exists():
            result[kind] = path
    return result


def format_top3(items: list[dict[str, Any]]) -> str:
    return " | ".join(f"{item['name']}:{float(item['confidence']):.6f}" for item in items)


def mock_prediction(image_code: str, views: list[str]) -> dict[str, Any]:
    orders = ["Diptera", "Hemiptera", "Hymenoptera", "Coleoptera"]
    digest = hashlib.sha256(image_code.encode("utf-8")).digest()
    index = digest[0] % len(orders)
    order = orders[index]
    confidence = 0.80 + ((digest[1] % 16) / 100.0)
    other = [value for value in orders if value != order][:2]
    top3 = [
        {"name": order, "confidence": confidence},
        {"name": other[0], "confidence": (1.0 - confidence) * 0.65},
        {"name": other[1], "confidence": (1.0 - confidence) * 0.35},
    ]
    return {
        "model": "mock_taxonomy",
        "kind": "mock",
        "fusion": "mock_four_view",
        "checkpoint": "",
        "views_used": views,
        "order": order,
        "order_confidence": confidence,
        "order_top3": top3,
        "species": "mock species",
        "species_confidence": confidence * 0.85,
        "species_top3": [{"name": "mock species", "confidence": confidence * 0.85}],
    }


def atomic_write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        with temp_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openpnp-root", type=Path, required=True)
    parser.add_argument("--plate", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--mock", action="store_true", help="Exercise integration without loading BioCLIP/checkpoints")
    args = parser.parse_args()

    root = args.openpnp_root.resolve()
    plate = normalize_plate(args.plate)
    csv_path = root / "Plate_spreadsheets" / f"{plate}.csv"
    image_dir = root / "Plate_insect_images" / f"P-{plate}"
    if not csv_path.exists():
        raise SystemExit(f"Plate CSV not found: {csv_path}")
    if not image_dir.exists():
        raise SystemExit(f"Plate image directory not found: {image_dir}")

    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    for column in AI_COLUMNS:
        if column not in fieldnames:
            fieldnames.append(column)

    classifier = None if args.mock else TaxonomyClassifier()
    processed = 0
    skipped = 0
    errors = 0
    run_records: list[dict[str, Any]] = []

    for row in rows:
        image_code = normalize_image_code(row.get("Image Code"))
        well = str(row.get("Well Number") or "").strip()
        if not image_code:
            skipped += 1
            continue
        if not args.force and str(row.get("AI Taxonomy Status") or "").strip().lower() == "classified":
            skipped += 1
            continue

        image_paths = find_image_paths(image_dir, image_code)
        try:
            if not image_paths:
                raise FileNotFoundError(f"No plate specimen images found for {image_code}")
            prediction = (
                mock_prediction(image_code, sorted(image_paths))
                if args.mock
                else classifier.classify(image_paths)  # type: ignore[union-attr]
            )
            row["AI Order"] = str(prediction["order"])
            row["AI Order Confidence"] = f"{float(prediction['order_confidence']):.6f}"
            row["AI Order Top 3"] = format_top3(prediction["order_top3"])
            row["AI Species"] = str(prediction.get("species", ""))
            row["AI Species Confidence"] = f"{float(prediction.get('species_confidence', 0.0)):.6f}"
            row["AI Species Top 3"] = format_top3(prediction.get("species_top3", []))
            row["AI Taxonomy Model"] = str(prediction["model"])
            row["AI Taxonomy Fusion"] = str(prediction["fusion"])
            row["AI Taxonomy Views"] = "+".join(prediction["views_used"])
            row["AI Taxonomy Status"] = "classified"
            row["AI Taxonomy Error"] = ""
            row["AI Taxonomy Timestamp"] = utc_now()
            processed += 1
            run_records.append({"plate": plate, "well": well, "image_code": image_code, **prediction})
            print(
                f"{image_code}: {prediction['order']} "
                f"({float(prediction['order_confidence']):.2%}) "
                f"views={'+'.join(prediction['views_used'])} model={prediction['model']}",
                flush=True,
            )
        except Exception as exc:
            errors += 1
            row["AI Taxonomy Status"] = "error"
            row["AI Taxonomy Error"] = str(exc)
            row["AI Taxonomy Timestamp"] = utc_now()
            run_records.append({
                "plate": plate,
                "well": well,
                "image_code": image_code,
                "status": "error",
                "error": str(exc),
            })
            print(f"ERROR {image_code}: {exc}", flush=True)

    atomic_write_csv(csv_path, fieldnames, rows)
    image_dir.mkdir(parents=True, exist_ok=True)
    last_run = image_dir / "taxonomy_predictions_last_run.jsonl"
    with last_run.open("w", encoding="utf-8") as handle:
        for record in run_records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    history = image_dir / "taxonomy_predictions_history.jsonl"
    with history.open("a", encoding="utf-8") as handle:
        for record in run_records:
            history_record = dict(record)
            history_record["recorded_at"] = utc_now()
            handle.write(json.dumps(history_record, sort_keys=True) + "\n")

    summary = {
        "plate": plate,
        "csv_path": str(csv_path),
        "image_dir": str(image_dir),
        "mock": bool(args.mock),
        "processed": processed,
        "skipped": skipped,
        "errors": errors,
        "rows": len(rows),
    }
    (image_dir / "taxonomy_predictions_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
