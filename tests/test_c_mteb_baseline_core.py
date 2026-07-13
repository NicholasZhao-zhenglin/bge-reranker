import unittest

from scripts.c_mteb_baseline_core import (
    aggregate_metrics,
    compare_metrics,
    rank_candidates,
    split_rows,
    summarize_ranking,
)


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

    def test_compare_metrics_reports_absolute_and_relative_change(self):
        baseline = {"mrr": 0.8, "hit_rate_at_1": 0.5, "test_queries": 100}
        candidate = {"mrr": 0.84, "hit_rate_at_1": 0.45, "test_queries": 100}

        result = compare_metrics(baseline, candidate)

        self.assertAlmostEqual(result["mrr"]["absolute_change"], 0.04)
        self.assertAlmostEqual(result["mrr"]["relative_change_percent"], 5.0)
        self.assertAlmostEqual(result["hit_rate_at_1"]["absolute_change"], -0.05)
        self.assertNotIn("test_queries", result)


if __name__ == "__main__":
    unittest.main()
