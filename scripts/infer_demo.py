import argparse
import time
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run a local BGE reranker inference demo")
    parser.add_argument(
        "--model",
        default=str(project_root / "models" / "bge-reranker-v2-m3"),
        help="Local model directory",
    )
    parser.add_argument("--query", default="什么是LoRA微调？")
    parser.add_argument(
        "--passage",
        default="北京今天天气不错",
    )
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()
    model_path = Path(args.model).expanduser()
    if not model_path.is_absolute():
        model_path = project_root / model_path
    model_path = model_path.resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Local model directory does not exist: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f"device: {device}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        torch_dtype=dtype,
        local_files_only=True,
    ).to(device)
    model.eval()

    encoded = tokenizer(
        args.query,
        args.passage,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_length,
    )
    encoded = {name: value.to(device) for name, value in encoded.items()}

    print(f"token_count: {encoded['input_ids'].shape[-1]}")
    print(f"query: {args.query}")
    print(f"passage: {args.passage}")

    start = time.perf_counter()
    with torch.inference_mode():
        logits = model(**encoded).logits.reshape(-1).float()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000

    raw_score = logits.item()
    probability = torch.sigmoid(logits).item()
    print(f"raw_score: {raw_score:.6f}")
    print(f"sigmoid_score: {probability:.6f}")
    print(f"latency_ms: {elapsed_ms:.2f}")


if __name__ == "__main__":
    main()
