"""Pure data and metric helpers for the C-MTEB reranking baseline."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable


def split_rows(rows: list[dict], train_size: int, validation_size: int, seed: int) -> dict[str, list[dict]]:
    """Deterministically split complete query rows without flattening candidates."""
    if train_size + validation_size >= len(rows):
        raise ValueError("train_size + validation_size must be smaller than the dataset size")

    import random

    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    train_end = train_size
    validation_end = train_size + validation_size
    return {
        "train": [rows[index] for index in indices[:train_end]],
        "validation": [rows[index] for index in indices[train_end:validation_end]],
        "test": [rows[index] for index in indices[validation_end:]],
    }


def validate_row(row: dict) -> None:
    """Validate the minimal C-MTEB reranking row contract."""
    required = {"query", "positive", "negative"}
    missing = required - row.keys()
    if missing:
        raise ValueError(f"missing fields: {sorted(missing)}")
    if not isinstance(row["query"], str) or not row["query"].strip():
        raise ValueError("query must be a non-empty string")
    if not isinstance(row["positive"], list) or not row["positive"]:
        raise ValueError("positive must be a non-empty list")
    if not isinstance(row["negative"], list) or not row["negative"]:
        raise ValueError("negative must be a non-empty list")
    overlap = set(row["positive"]) & set(row["negative"])
    if overlap:
        raise ValueError(f"positive and negative overlap: {sorted(overlap)[:3]}")


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def rank_candidates(row: dict, scores: list[float]) -> list[dict]:
    candidates = row["positive"] + row["negative"]
    if len(candidates) != len(scores):
        raise ValueError("candidate and score counts differ")
    positive = set(row["positive"])
    ranked = sorted(
        (
            {"passage": passage, "score": float(score), "is_positive": passage in positive, "original_index": index}
            for index, (passage, score) in enumerate(zip(candidates, scores))
        ),
        key=lambda item: (-item["score"], item["original_index"]),
    )
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank
    return ranked


def summarize_ranking(ranked: list[dict], cutoffs: tuple[int, ...] = (1, 3, 5, 10)) -> dict[str, float]:
    positive_ranks = [item["rank"] for item in ranked if item["is_positive"]]
    if not positive_ranks:
        raise ValueError("ranking contains no positive candidate")
    first_rank = min(positive_ranks)
    result = {"mrr": 1.0 / first_rank}
    for cutoff in cutoffs:
        hits = sum(1 for item in ranked[:cutoff] if item["is_positive"])
        result[f"recall_at_{cutoff}"] = hits / len(positive_ranks)
        result[f"hit_rate_at_{cutoff}"] = float(hits > 0)
        dcg = sum(1.0 / math.log2(item["rank"] + 1) for item in ranked[:cutoff] if item["is_positive"])
        ideal_hits = min(len(positive_ranks), cutoff)
        ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
        result[f"ndcg_at_{cutoff}"] = dcg / ideal_dcg if ideal_dcg else 0.0
    return result


def aggregate_metrics(per_query: list[dict[str, float]]) -> dict[str, float]:
    if not per_query:
        raise ValueError("cannot aggregate an empty evaluation")
    keys = per_query[0].keys()
    return {key: sum(row[key] for row in per_query) / len(per_query) for key in keys}


def compare_metrics(baseline: dict, candidate: dict) -> dict[str, dict[str, float]]:
    """Compare shared ranking metrics while ignoring counts and runtime metadata."""
    metric_prefixes = ("mrr", "recall_at_", "hit_rate_at_", "ndcg_at_")
    comparison: dict[str, dict[str, float]] = {}
    for key in baseline.keys() & candidate.keys():
        if not key.startswith(metric_prefixes):
            continue
        baseline_value = float(baseline[key])
        candidate_value = float(candidate[key])
        absolute_change = candidate_value - baseline_value
        relative_change = absolute_change / baseline_value * 100 if baseline_value else 0.0
        comparison[key] = {
            "baseline": baseline_value,
            "lora": candidate_value,
            "absolute_change": absolute_change,
            "relative_change_percent": relative_change,
        }
    return dict(sorted(comparison.items()))
