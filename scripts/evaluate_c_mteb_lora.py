"""Evaluate a trained LoRA reranker on the frozen C-MTEB test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from c_mteb_baseline_core import compare_metrics, write_jsonl
from evaluate_c_mteb_baseline import RESULTS, ROOT, evaluate_test, prepare_splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=str(ROOT / "models" / "bge-reranker-v2-m3"))
    parser.add_argument(
        "--adapter",
        default=str(ROOT / "outputs" / "c-mteb-lora-epoch1" / "adapter"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs" / "c-mteb-lora-epoch1" / "test_evaluation"),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--train-size", type=int, default=4000)
    parser.add_argument("--validation-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-resplit", action="store_true")
    args = parser.parse_args()

    base_model_path = Path(args.base_model).expanduser().resolve()
    adapter_path = Path(args.adapter).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    baseline_path = RESULTS / "baseline_metrics.json"
    for path, description in (
        (base_model_path, "base model"),
        (adapter_path, "LoRA adapter"),
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"Local {description} directory does not exist: {path}")
    if not baseline_path.is_file():
        raise FileNotFoundError(f"Baseline metrics do not exist: {baseline_path}")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    splits = prepare_splits(args)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if len(splits["test"]) != baseline.get("test_queries"):
        raise ValueError("Frozen test split size differs from the saved baseline")
    if args.max_length != baseline.get("max_length"):
        raise ValueError(
            f"max_length must match baseline ({baseline.get('max_length')}) for a fair comparison"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_path, dtype=dtype, local_files_only=True
    )
    model = PeftModel.from_pretrained(base_model, adapter_path, local_files_only=True).to(device)
    model.eval()

    metrics, rankings = evaluate_test(
        model, tokenizer, splits["test"], device, args.batch_size, args.max_length
    )
    metrics.update(
        {
            "base_model_path": str(base_model_path),
            "adapter_path": str(adapter_path),
            "device": str(device),
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "seed": args.seed,
            "split_sizes": {name: len(rows) for name, rows in splits.items()},
        }
    )
    comparison = compare_metrics(baseline, metrics)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "lora_test_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "baseline_vs_lora.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_jsonl(output_dir / "lora_test_rankings.jsonl", rankings)
    print(json.dumps({"lora": metrics, "comparison": comparison}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
