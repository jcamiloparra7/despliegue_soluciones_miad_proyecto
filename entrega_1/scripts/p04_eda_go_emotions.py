#!/usr/bin/env python3
"""Generate EDA artifacts for P04 using GoEmotions simplified splits."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter
from itertools import combinations
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


def percentile(values: list[int], q: float) -> float:
    if not values:
        return float("nan")
    if q <= 0:
        return float(min(values))
    if q >= 100:
        return float(max(values))
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (q / 100)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_vals[int(k)])
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return float(d0 + d1)


def text_stats(texts: list[str]) -> dict:
    char_lens = [len(t) for t in texts]
    word_lens = [len(t.split()) for t in texts]
    return {
        "char_len_mean": round(statistics.fmean(char_lens), 4),
        "char_len_median": statistics.median(char_lens),
        "char_len_p95": round(percentile(char_lens, 95), 4),
        "char_len_min": min(char_lens),
        "char_len_max": max(char_lens),
        "word_len_mean": round(statistics.fmean(word_lens), 4),
        "word_len_median": statistics.median(word_lens),
        "word_len_p95": round(percentile(word_lens, 95), 4),
        "word_len_min": min(word_lens),
        "word_len_max": max(word_lens),
    }


def read_split(split_path: Path) -> tuple[list[str], list[list[int]], list[str]]:
    table = pq.read_table(split_path, columns=["text", "labels", "id"])
    texts = table["text"].to_pylist()
    labels = table["labels"].to_pylist()
    ids = table["id"].to_pylist()
    labels = [l or [] for l in labels]
    return texts, labels, ids


def summarize_split(texts: list[str], labels: list[list[int]], ids: list[str]) -> tuple[dict, Counter]:
    label_counts = Counter()
    labels_per_sample = [len(l) for l in labels]
    for row in labels:
        for idx in row:
            if 0 <= idx < len(EMOTION_NAMES):
                label_counts[EMOTION_NAMES[idx]] += 1
            else:
                label_counts[f"unknown_{idx}"] += 1

    unique_ids = len(set(ids))
    duplicate_ids = len(ids) - unique_ids
    stats = text_stats(texts)
    summary = {
        "rows": len(texts),
        "unique_ids": unique_ids,
        "duplicate_ids": duplicate_ids,
        "avg_labels_per_sample": round(statistics.fmean(labels_per_sample), 6),
        "median_labels_per_sample": statistics.median(labels_per_sample),
        "max_labels_per_sample": max(labels_per_sample),
        "text_stats": stats,
    }
    return summary, label_counts


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    data_root = repo_root / "data" / "go_emotions" / "simplified"
    out_dir = repo_root / "entrega_1" / "artifacts" / "p04"
    out_dir.mkdir(parents=True, exist_ok=True)

    split_paths = {
        "train": data_root / "train-00000-of-00001.parquet",
        "validation": data_root / "validation-00000-of-00001.parquet",
        "test": data_root / "test-00000-of-00001.parquet",
    }

    split_data = {}
    split_summaries = {}
    split_label_counts = {}

    for split, path in split_paths.items():
        texts, labels, ids = read_split(path)
        split_data[split] = {"texts": texts, "labels": labels, "ids": ids}
        summary, label_counts = summarize_split(texts, labels, ids)
        split_summaries[split] = summary
        split_label_counts[split] = label_counts

    # ID overlap checks between splits (to check leakage).
    train_ids = set(split_data["train"]["ids"])
    val_ids = set(split_data["validation"]["ids"])
    test_ids = set(split_data["test"]["ids"])

    overlap = {
        "train_validation_overlap": len(train_ids & val_ids),
        "train_test_overlap": len(train_ids & test_ids),
        "validation_test_overlap": len(val_ids & test_ids),
    }

    # Global label frequency.
    total_label_counts = Counter()
    for c in split_label_counts.values():
        total_label_counts.update(c)
    total_assigned_labels = sum(total_label_counts.values())

    # Co-occurrence pairs on combined splits.
    pair_counts = Counter()
    for split in split_data.values():
        for row in split["labels"]:
            uniq = sorted(set(row))
            for i, j in combinations(uniq, 2):
                pair_counts[(EMOTION_NAMES[i], EMOTION_NAMES[j])] += 1

    top_pairs = [
        {"emotion_a": a, "emotion_b": b, "cooccurrence_count": c}
        for (a, b), c in pair_counts.most_common(30)
    ]

    # Write emotion counts by split (+ total).
    emotion_rows = []
    for emotion in EMOTION_NAMES:
        row = {"emotion": emotion}
        total = 0
        for split in ["train", "validation", "test"]:
            count = split_label_counts[split].get(emotion, 0)
            row[f"{split}_count"] = count
            total += count
        row["total_count"] = total
        row["total_percentage_of_assigned_labels"] = (
            round((total / total_assigned_labels) * 100, 4) if total_assigned_labels else 0.0
        )
        emotion_rows.append(row)
    emotion_rows.sort(key=lambda r: r["total_count"], reverse=True)

    write_csv(
        out_dir / "p04_emotion_counts_by_split.csv",
        emotion_rows,
        [
            "emotion",
            "train_count",
            "validation_count",
            "test_count",
            "total_count",
            "total_percentage_of_assigned_labels",
        ],
    )

    # Write text length stats CSV.
    text_stat_rows = []
    for split in ["train", "validation", "test"]:
        s = split_summaries[split]
        ts = s["text_stats"]
        text_stat_rows.append(
            {
                "split": split,
                "rows": s["rows"],
                "avg_labels_per_sample": s["avg_labels_per_sample"],
                "char_len_mean": ts["char_len_mean"],
                "char_len_median": ts["char_len_median"],
                "char_len_p95": ts["char_len_p95"],
                "word_len_mean": ts["word_len_mean"],
                "word_len_median": ts["word_len_median"],
                "word_len_p95": ts["word_len_p95"],
                "duplicate_ids": s["duplicate_ids"],
            }
        )

    write_csv(
        out_dir / "p04_text_length_stats.csv",
        text_stat_rows,
        [
            "split",
            "rows",
            "avg_labels_per_sample",
            "char_len_mean",
            "char_len_median",
            "char_len_p95",
            "word_len_mean",
            "word_len_median",
            "word_len_p95",
            "duplicate_ids",
        ],
    )

    write_csv(
        out_dir / "p04_top_label_cooccurrence_pairs.csv",
        top_pairs,
        ["emotion_a", "emotion_b", "cooccurrence_count"],
    )

    summary = {
        "split_summaries": split_summaries,
        "id_overlap": overlap,
        "total_simplified_rows": sum(split_summaries[s]["rows"] for s in split_summaries),
        "total_assigned_labels": total_assigned_labels,
        "avg_labels_per_sample_global": round(
            total_assigned_labels / sum(split_summaries[s]["rows"] for s in split_summaries), 6
        ),
        "top_10_emotions_by_count": [
            {"emotion": e, "count": c}
            for e, c in total_label_counts.most_common(10)
        ],
        "top_10_label_cooccurrence_pairs": top_pairs[:10],
    }
    (out_dir / "p04_eda_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Generated: {out_dir / 'p04_eda_summary.json'}")
    print(f"Generated: {out_dir / 'p04_emotion_counts_by_split.csv'}")
    print(f"Generated: {out_dir / 'p04_text_length_stats.csv'}")
    print(f"Generated: {out_dir / 'p04_top_label_cooccurrence_pairs.csv'}")


if __name__ == "__main__":
    main()
