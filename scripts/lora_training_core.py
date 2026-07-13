"""Pure helpers shared by the LoRA training entry point and its tests."""

from __future__ import annotations

import math


def cast_trainable_parameters_to_fp32(model) -> None:
    """Keep frozen base weights compact while making scaled gradients safe."""
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()


def flatten_rows(rows: list[dict]) -> list[tuple[str, str, float]]:
    """Convert grouped reranking rows into labeled query-passage pairs."""
    pairs: list[tuple[str, str, float]] = []
    for row in rows:
        query = row["query"]
        pairs.extend((query, passage, 1.0) for passage in row["positive"])
        pairs.extend((query, passage, 0.0) for passage in row["negative"])
    return pairs


def optimizer_steps_per_epoch(
    pair_count: int, batch_size: int, gradient_accumulation_steps: int
) -> int:
    micro_batches = math.ceil(pair_count / batch_size)
    return math.ceil(micro_batches / gradient_accumulation_steps)


def loss_trend(history: list[float], window: int) -> dict[str, float] | None:
    """Compare the first and most recent complete loss windows."""
    if len(history) < window * 2:
        return None
    early_average = sum(history[:window]) / window
    recent_average = sum(history[-window:]) / window
    relative_change = (recent_average - early_average) / max(abs(early_average), 1e-12)
    return {
        "early_average": early_average,
        "recent_average": recent_average,
        "relative_change": relative_change,
    }
