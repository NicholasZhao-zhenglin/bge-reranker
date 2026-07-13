import unittest

from scripts.c_mteb_baseline_core import aggregate_metrics, rank_candidates, split_rows, summarize_ranking


class CMTEDBaselineCoreTests(unittest.TestCase):
    def test_split_is_deterministic_and_keeps_complete_rows(self):
        rows = [{"query": str(index), "positive": ["p"], "negative": ["n"]} for index in range(10)]
        first = split_rows(rows, train_size=6, validation_size=2, seed=42)
        second = split_rows(rows, train_size=6, validation_size=2, seed=42)
        self.assertEqual(first, second)
        self.assertEqual({len(value) for value in first.values()}, {6, 2, 2})

    def test_ranking_metrics_use_all_positive_candidates(self):
        row = {"query": "q", "positive": ["a", "b"], "negative": ["x", "y", "z"]}
        ranked = rank_candidates(row, [0.7, 0.9, 0.8, 0.2, 0.0])
        metrics = summarize_ranking(ranked)
        self.assertEqual([item["passage"] for item in ranked[:3]], ["b", "x", "a"])
        self.assertEqual(metrics["recall_at_1"], 0.5)
        self.assertEqual(metrics["hit_rate_at_1"], 1.0)
        self.assertEqual(metrics["recall_at_3"], 1.0)
        self.assertEqual(metrics["mrr"], 1.0)

    def test_aggregate_metrics(self):
        result = aggregate_metrics([{"mrr": 1.0, "recall_at_1": 0.0}, {"mrr": 0.5, "recall_at_1": 1.0}])
        self.assertEqual(result, {"mrr": 0.75, "recall_at_1": 0.5})


if __name__ == "__main__":
    unittest.main()
