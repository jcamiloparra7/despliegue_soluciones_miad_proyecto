#!/usr/bin/env python3
"""Generate reproducible profiling artifacts for P02 (data + characterization)."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


EMOTION_NAMES = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
]


def summarize_parquet(path: Path) -> dict:
    pf = pq.ParquetFile(path)
    schema = [{"name": f.name, "type": str(f.type)} for f in pf.schema_arrow]
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "rows": pf.metadata.num_rows,
        "columns": len(schema),
        "schema": schema,
    }


def analyze_simplified_labels(split_path: Path) -> tuple[Counter, Counter]:
    table = pq.read_table(split_path, columns=["labels"])
    labels_array = table["labels"]

    label_counts: Counter = Counter()
    labels_per_sample: Counter = Counter()

    for row in labels_array.to_pylist():
        row_labels = row or []
        labels_per_sample[len(row_labels)] += 1
        for label_id in row_labels:
            if 0 <= label_id < len(EMOTION_NAMES):
                label_counts[EMOTION_NAMES[label_id]] += 1
            else:
                label_counts[f"unknown_{label_id}"] += 1

    return label_counts, labels_per_sample


def count_text_quality(split_path: Path) -> dict:
    table = pq.read_table(split_path, columns=["text"])
    texts = table["text"].to_pylist()

    null_count = sum(1 for t in texts if t is None)
    empty_count = sum(1 for t in texts if isinstance(t, str) and not t.strip())
    return {
        "null_text_count": null_count,
        "empty_or_whitespace_text_count": empty_count,
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    data_root = repo_root / "data" / "go_emotions"
    out_dir = repo_root / "entrega_1" / "artifacts" / "p02"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_train = data_root / "raw" / "train-00000-of-00001.parquet"
    simp_train = data_root / "simplified" / "train-00000-of-00001.parquet"
    simp_val = data_root / "simplified" / "validation-00000-of-00001.parquet"
    simp_test = data_root / "simplified" / "test-00000-of-00001.parquet"

    datasets = {
        "raw_train": raw_train,
        "simplified_train": simp_train,
        "simplified_validation": simp_val,
        "simplified_test": simp_test,
    }

    profile = {"datasets": {}, "quality": {}, "derived": {}}

    missing_files = [
    str(path) for path in datasets.values() if not path.is_file()
]

    if missing_files:
        raise FileNotFoundError(
            "No se encontraron los siguientes archivos de GoEmotions:\n"
            + "\n".join(missing_files)
        )

    for name, path in datasets.items():
        profile["datasets"][name] = summarize_parquet(path)
        profile["quality"][name] = count_text_quality(path)

    # Aggregated label stats over simplified splits.
    total_label_counts: Counter = Counter()
    total_labels_per_sample: Counter = Counter()
    split_label_summary = {}
    split_labels_per_sample_summary = {}

    for split_name, split_path in [
        ("train", simp_train),
        ("validation", simp_val),
        ("test", simp_test),
    ]:
        split_label_counts, split_labels_per_sample = analyze_simplified_labels(split_path)
        split_label_summary[split_name] = dict(split_label_counts)
        split_labels_per_sample_summary[split_name] = dict(split_labels_per_sample)

        total_label_counts.update(split_label_counts)
        total_labels_per_sample.update(split_labels_per_sample)

    total_simplified_rows = (
        profile["datasets"]["simplified_train"]["rows"]
        + profile["datasets"]["simplified_validation"]["rows"]
        + profile["datasets"]["simplified_test"]["rows"]
    )
    total_assigned_labels = sum(total_label_counts.values())

    profile["derived"] = {
        "total_simplified_rows": total_simplified_rows,
        "total_assigned_labels": total_assigned_labels,
        "avg_labels_per_sample": round(total_assigned_labels / total_simplified_rows, 6),
        "label_counts_by_split": split_label_summary,
        "labels_per_sample_by_split": split_labels_per_sample_summary,
        "label_counts_total": dict(total_label_counts),
        "labels_per_sample_total": dict(total_labels_per_sample),
    }

    # Save JSON profile.
    profile_path = out_dir / "p02_dataset_profile.json"
    profile_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")

    # Save emotion distribution CSV.
    emotion_rows = []
    for emotion in EMOTION_NAMES:
        count = total_label_counts.get(emotion, 0)
        pct = (count / total_assigned_labels * 100) if total_assigned_labels else 0
        emotion_rows.append(
            {"emotion": emotion, "count": count, "percentage_of_assigned_labels": round(pct, 4)}
        )
    emotion_rows.sort(key=lambda r: r["count"], reverse=True)
    write_csv(
        out_dir / "p02_emotion_distribution.csv",
        emotion_rows,
        ["emotion", "count", "percentage_of_assigned_labels"],
    )

    # Save labels-per-sample distribution CSV.
    lps_rows = []
    for k in sorted(total_labels_per_sample):
        c = total_labels_per_sample[k]
        pct = (c / total_simplified_rows * 100) if total_simplified_rows else 0
        lps_rows.append(
            {
                "labels_per_sample": k,
                "count": c,
                "percentage_of_samples": round(pct, 4),
            }
        )
    write_csv(
        out_dir / "p02_labels_per_sample_distribution.csv",
        lps_rows,
        ["labels_per_sample", "count", "percentage_of_samples"],
    )

    print(f"Generated: {profile_path}")
    print(f"Generated: {out_dir / 'p02_emotion_distribution.csv'}")
    print(f"Generated: {out_dir / 'p02_labels_per_sample_distribution.csv'}")


if __name__ == "__main__":
    main()
