import unittest

from scripts.lora_training_core import flatten_rows, loss_trend, optimizer_steps_per_epoch


class LoRATrainingCoreTests(unittest.TestCase):
    def test_flatten_rows_creates_labeled_pairs(self):
        rows = [
            {
                "query": "q",
                "positive": ["p1", "p2"],
                "negative": ["n1"],
            }
        ]
        pairs = flatten_rows(rows)
        self.assertEqual(
            pairs,
            [("q", "p1", 1.0), ("q", "p2", 1.0), ("q", "n1", 0.0)],
        )

    def test_optimizer_steps_accounts_for_gradient_accumulation(self):
        self.assertEqual(
            optimizer_steps_per_epoch(pair_count=66_214, batch_size=8, gradient_accumulation_steps=2),
            4_139,
        )

    def test_loss_trend_compares_early_and_recent_windows(self):
        history = [1.0] * 20 + [0.5] * 20
        trend = loss_trend(history, window=20)
        self.assertEqual(trend["early_average"], 1.0)
        self.assertEqual(trend["recent_average"], 0.5)
        self.assertEqual(trend["relative_change"], -0.5)


if __name__ == "__main__":
    unittest.main()
