"""Build fixed C-MTEB splits and evaluate the local BGE reranker on test."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from c_mteb_baseline_core import (
    aggregate_metrics,
    rank_candidates,
    read_jsonl,
    split_rows,
    summarize_ranking,
    validate_row,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "external" / "c_mteb_t2reranking" / "dev-00000-of-00001.parquet"
RESULTS = ROOT / "data" / "experiments" / "c_mteb_t2reranking"


def load_source_rows() -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "Missing dependency 'datasets'. Activate the bge_reranker conda environment first."
        ) from error
    dataset = load_dataset("parquet", data_files=str(SOURCE), split="train")
    return dataset.to_list()


def prepare_splits(args: argparse.Namespace) -> dict[str, list[dict]]:
    split_paths = {name: RESULTS / f"{name}.jsonl" for name in ("train", "validation", "test")}
    can_reuse = not args.force_resplit and all(path.is_file() for path in split_paths.values())
    if can_reuse:
        splits = {name: read_jsonl(path) for name, path in split_paths.items()}
        can_reuse = all(
            all(bool(row.get("positive")) and bool(row.get("negative")) for row in rows)
            for rows in splits.values()
        )
    if can_reuse:
        pass
    else:
        source_rows = load_source_rows()
        if len(source_rows) != 6129:
            raise ValueError(f"Unexpected source row count: {len(source_rows)}; expected 6129")
        rows = [row for row in source_rows if row["positive"] and row["negative"]]
        filtered_count = len(source_rows) - len(rows)
        if args.train_size + args.validation_size >= len(rows):
            raise ValueError("requested train/validation sizes leave no test rows after filtering")
        splits = split_rows(rows, args.train_size, args.validation_size, args.seed)
        for name, rows_for_split in splits.items():
            write_jsonl(split_paths[name], rows_for_split)
        (RESULTS / "split_manifest.json").write_text(
            json.dumps(
                {
                    "source": str(SOURCE),
                    "seed": args.seed,
                    "source_rows": len(source_rows),
                    "filtered_rows_without_both_classes": filtered_count,
                    "filter_policy": "keep only rows with at least one positive and one negative",
                    "train_size": len(splits["train"]),
                    "validation_size": len(splits["validation"]),
                    "test_size": len(splits["test"]),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    for rows in splits.values():
        for row in rows:
            validate_row(row)
    return splits


def score_pairs(model, tokenizer, pairs, device, batch_size: int, max_length: int) -> list[float]:
    import torch

    scores: list[float] = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        inputs = tokenizer(
            [query for query, _ in batch],
            [passage for _, passage in batch],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = {name: value.to(device) for name, value in inputs.items()}
        with torch.inference_mode():
            batch_scores = model(**inputs).logits.reshape(-1).float().cpu().tolist()
        scores.extend(batch_scores)
    return scores


def evaluate_test(model, tokenizer, rows, device, batch_size: int, max_length: int) -> tuple[dict, list[dict]]:
    pairs = [(row["query"], passage) for row in rows for passage in row["positive"] + row["negative"]]
    started_at = time.perf_counter()
    scores = score_pairs(model, tokenizer, pairs, device, batch_size, max_length)
    if device.type == "cuda":
        import torch

        torch.cuda.synchronize()
    elapsed_seconds = time.perf_counter() - started_at

    rankings: list[dict] = []
    per_query_metrics: list[dict[str, float]] = []
    score_index = 0
    for query_index, row in enumerate(rows):
        candidate_count = len(row["positive"]) + len(row["negative"])
        ranked = rank_candidates(row, scores[score_index : score_index + candidate_count])
        score_index += candidate_count
        per_query_metrics.append(summarize_ranking(ranked))
        rankings.append(
            {
                "query_index": query_index,
                "query": row["query"],
                "positive_count": len(row["positive"]),
                "negative_count": len(row["negative"]),
                "metrics": per_query_metrics[-1],
                "ranked_candidates": ranked,
            }
        )

    metrics = aggregate_metrics(per_query_metrics)
    metrics.update(
        {
            "test_queries": len(rows),
            "candidate_pairs": len(pairs),
            "elapsed_seconds": round(elapsed_seconds, 3),
            "pairs_per_second": round(len(pairs) / elapsed_seconds, 2),
        }
    )
    return metrics, rankings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(ROOT / "models" / "bge-reranker-v2-m3"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--train-size", type=int, default=4000)
    parser.add_argument("--validation-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-resplit", action="store_true")
    args = parser.parse_args()

    model_path = Path(args.model).expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Local model directory does not exist: {model_path}")
    if not SOURCE.is_file():
        raise FileNotFoundError(f"C-MTEB parquet file does not exist: {SOURCE}")

    splits = prepare_splits(args)
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, dtype=dtype, local_files_only=True
    ).to(device)
    model.eval()

    metrics, rankings = evaluate_test(
        model, tokenizer, splits["test"], device, args.batch_size, args.max_length
    )
    metrics.update(
        {
            "model_path": str(model_path),
            "device": str(device),
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "seed": args.seed,
            "split_sizes": {name: len(rows) for name, rows in splits.items()},
        }
    )
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "baseline_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_jsonl(RESULTS / "baseline_rankings.jsonl", rankings)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
