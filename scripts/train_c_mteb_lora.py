"""Fine-tune the local BGE reranker with LoRA and live resource monitoring."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import psutil
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup

from c_mteb_baseline_core import aggregate_metrics, rank_candidates, read_jsonl, summarize_ranking
from lora_training_core import (
    cast_trainable_parameters_to_fp32,
    flatten_rows,
    loss_trend,
    optimizer_steps_per_epoch,
)


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = ROOT / "data" / "experiments" / "c_mteb_t2reranking"


class PairDataset(Dataset):
    def __init__(self, pairs: list[tuple[str, str, float]]) -> None:
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[str, str, float]:
        return self.pairs[index]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_collate(tokenizer, max_length: int):
    def collate(batch: list[tuple[str, str, float]]) -> dict[str, torch.Tensor]:
        encoded = tokenizer(
            [query for query, _, _ in batch],
            [passage for _, passage, _ in batch],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor([label for _, _, label in batch], dtype=torch.float32)
        return encoded

    return collate


def emit_event(log_path: Path, event: dict) -> None:
    event = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **event}
    with log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event, ensure_ascii=False) + "\n")


def resource_snapshot(device: torch.device) -> dict[str, float]:
    memory = psutil.virtual_memory()
    snapshot = {
        "ram_used_gb": round(memory.used / 1024**3, 2),
        "ram_percent": round(memory.percent, 1),
    }
    if device.type == "cuda":
        free_bytes, _ = torch.cuda.mem_get_info()
        snapshot.update(
            {
                "gpu_allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
                "gpu_reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 2),
                "gpu_peak_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
                "gpu_free_gb": round(free_bytes / 1024**3, 2),
            }
        )
    return snapshot


def score_validation_pairs(model, tokenizer, pairs, device, batch_size: int, max_length: int) -> list[float]:
    scores: list[float] = []
    model.eval()
    for start in tqdm(range(0, len(pairs), batch_size), desc="validation", leave=False):
        batch = pairs[start : start + batch_size]
        encoded = tokenizer(
            [query for query, _ in batch],
            [passage for _, passage in batch],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {name: value.to(device) for name, value in encoded.items()}
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            logits = model(**encoded).logits.reshape(-1)
        scores.extend(logits.float().cpu().tolist())
    return scores


def evaluate(model, tokenizer, rows, device, batch_size: int, max_length: int) -> dict[str, float]:
    pairs = [(row["query"], passage) for row in rows for passage in row["positive"] + row["negative"]]
    scores = score_validation_pairs(model, tokenizer, pairs, device, batch_size, max_length)
    per_query: list[dict[str, float]] = []
    score_index = 0
    for row in rows:
        candidate_count = len(row["positive"]) + len(row["negative"])
        ranked = rank_candidates(row, scores[score_index : score_index + candidate_count])
        score_index += candidate_count
        per_query.append(summarize_ranking(ranked))
    return aggregate_metrics(per_query)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(ROOT / "models" / "bge-reranker-v2-m3"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "c-mteb-lora-epoch1"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--validation-batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-pairs", type=int)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    model_path = Path(args.model).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training_metrics.jsonl"
    log_path.unlink(missing_ok=True)

    train_rows = read_jsonl(EXPERIMENT_DIR / "train.jsonl")
    validation_rows = read_jsonl(EXPERIMENT_DIR / "validation.jsonl")
    train_pairs = flatten_rows(train_rows)
    random.Random(args.seed).shuffle(train_pairs)
    if args.max_train_pairs:
        train_pairs = train_pairs[: args.max_train_pairs]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This training configuration requires a CUDA GPU")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, dtype=torch.float16, local_files_only=True
    ).to(device)
    model = get_peft_model(
        model,
        LoraConfig(
            task_type="SEQ_CLS",
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["query", "key", "value"],
            bias="none",
        ),
    )
    cast_trainable_parameters_to_fp32(model)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.train()

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    steps_per_epoch = optimizer_steps_per_epoch(
        len(train_pairs), args.batch_size, args.gradient_accumulation_steps
    )
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, math.ceil(total_steps * args.warmup_ratio))

    loader = DataLoader(
        PairDataset(train_pairs),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=make_collate(tokenizer, args.max_length),
        generator=torch.Generator().manual_seed(args.seed),
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler("cuda")

    config = {
        **vars(args),
        "model": str(model_path),
        "output_dir": str(output_dir),
        "train_queries": len(train_rows),
        "train_pairs": len(train_pairs),
        "validation_queries": len(validation_rows),
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "trainable_parameters": trainable,
        "total_parameters": total_parameters,
    }
    (output_dir / "training_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)
    emit_event(log_path, {"event": "start", **config, **resource_snapshot(device)})

    global_step = 0
    loss_history: list[float] = []
    started_at = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(total=total_steps, desc="LoRA training", dynamic_ncols=True)

    try:
        for epoch in range(args.epochs):
            accumulated_losses: list[float] = []
            for micro_step, batch in enumerate(loader, start=1):
                batch = {name: value.to(device) for name, value in batch.items()}
                labels = batch.pop("labels")
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(**batch).logits.reshape(-1)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        logits.float(), labels
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at micro step {micro_step}: {loss}")
                accumulated_losses.append(float(loss.detach().cpu()))
                scaler.scale(loss / args.gradient_accumulation_steps).backward()

                is_update = (
                    micro_step % args.gradient_accumulation_steps == 0
                    or micro_step == len(loader)
                )
                if not is_update:
                    continue

                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                update_loss = sum(accumulated_losses) / len(accumulated_losses)
                accumulated_losses.clear()
                loss_history.append(update_loss)
                progress.update(1)
                progress.set_postfix(loss=f"{update_loss:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

                if global_step == 1 or global_step % args.log_every == 0 or global_step == total_steps:
                    elapsed = time.perf_counter() - started_at
                    recent = loss_history[-args.log_every :]
                    snapshot = resource_snapshot(device)
                    event = {
                        "event": "train_step",
                        "epoch": epoch + 1,
                        "step": global_step,
                        "total_steps": total_steps,
                        "loss": round(update_loss, 6),
                        "loss_moving_average": round(sum(recent) / len(recent), 6),
                        "learning_rate": scheduler.get_last_lr()[0],
                        "grad_norm": round(float(grad_norm), 6),
                        "pairs_per_second": round(global_step * args.batch_size * args.gradient_accumulation_steps / elapsed, 2),
                        "elapsed_minutes": round(elapsed / 60, 2),
                        "eta_minutes": round((elapsed / global_step) * (total_steps - global_step) / 60, 2),
                        **snapshot,
                    }
                    emit_event(log_path, event)
                    tqdm.write(
                        "step={step}/{total_steps} loss={loss:.4f} avg={loss_moving_average:.4f} "
                        "lr={learning_rate:.2e} grad={grad_norm:.3f} gpu={gpu_allocated_gb:.2f}GB "
                        "free={gpu_free_gb:.2f}GB ram={ram_percent:.1f}% eta={eta_minutes:.1f}m".format(**event)
                    )
                    if snapshot["ram_percent"] >= 93:
                        raise MemoryError(f"System RAM reached {snapshot['ram_percent']}%")
                    trend = loss_trend(loss_history, window=50)
                    if trend and trend["relative_change"] > 0.5:
                        raise FloatingPointError(
                            f"Loss diverged: early={trend['early_average']:.4f}, "
                            f"recent={trend['recent_average']:.4f}"
                        )

        progress.close()
        model.save_pretrained(output_dir / "adapter")
        tokenizer.save_pretrained(output_dir / "adapter")
        validation_metrics = None
        if not args.skip_validation:
            validation_metrics = evaluate(
                model,
                tokenizer,
                validation_rows,
                device,
                args.validation_batch_size,
                args.max_length,
            )
            print("validation=" + json.dumps(validation_metrics, ensure_ascii=False), flush=True)
            emit_event(log_path, {"event": "validation", **validation_metrics})

        trend = loss_trend(loss_history, window=min(50, max(1, len(loss_history) // 2)))
        summary = {
            "status": "completed",
            "steps": global_step,
            "final_loss": loss_history[-1],
            "loss_trend": trend,
            "validation": validation_metrics,
            "elapsed_minutes": round((time.perf_counter() - started_at) / 60, 2),
            **resource_snapshot(device),
        }
        (output_dir / "training_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        emit_event(log_path, {"event": "complete", **summary})
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    except Exception as error:
        progress.close()
        event = {
            "event": "error",
            "error_type": type(error).__name__,
            "message": str(error),
            "step": global_step,
            **resource_snapshot(device),
        }
        emit_event(log_path, event)
        print(json.dumps(event, ensure_ascii=False, indent=2), flush=True)
        raise


if __name__ == "__main__":
    main()
