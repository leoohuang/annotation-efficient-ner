import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_data import duplicate_groups, split_indices, to_spans
from score_spans import score


class DataTests(unittest.TestCase):
    def test_near_duplicates_form_transitive_groups(self):
        rows = [{"tokens": text.split()} for text in [
            "a b c d e", "a b c d e f", "b c d e f"]]
        groups, pairs = duplicate_groups(rows)
        self.assertEqual(groups, [[0, 1, 2]])
        self.assertEqual(len(pairs), 2)

    def test_duplicates_stay_together_and_all_records_are_assigned(self):
        rows = [{"tokens": [f"word{i}"]} for i in range(20)]
        rows += [{"tokens": ["WORD3"]}]
        splits = split_indices(rows)
        assigned = [i for indices in splits.values() for i in indices]
        self.assertEqual(sorted(assigned), list(range(21)))
        self.assertTrue(any(3 in indices and 20 in indices for indices in splits.values()))
        self.assertEqual(splits, split_indices(rows))

    def test_nested_spans_are_preserved(self):
        row = {"text": "air filters", "tokens": ["air", "filters"], "entities": [
            {"start": 0, "end": 2, "type": "PhysicalObject/A"},
            {"start": 1, "end": 2, "type": "PhysicalObject/A"}]}
        self.assertEqual(len(to_spans(row, 0)["entities"]), 2)


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.gold = [{"record_index": 0, "tokens": ["air", "filters"], "entities": [
            {"start": 0, "end": 2, "type": "PhysicalObject"},
            {"start": 1, "end": 2, "type": "PhysicalObject"}]}]

    def test_oracle_counts_both_nested_entities(self):
        result = score(self.gold, self.gold)["micro"]
        self.assertEqual((result["tp"], result["fp"], result["fn"], result["f1"]), (2, 0, 0, 1))

    def test_missing_inner_span_is_false_negative(self):
        pred = [{"record_index": 0, "entities": self.gold[0]["entities"][:1]}]
        result = score(self.gold, pred)["micro"]
        self.assertEqual(result["fn"], 1)
        self.assertAlmostEqual(result["f1"], 2 / 3)

    def test_wrong_type_counts_as_false_positive_and_false_negative(self):
        pred = [{"record_index": 0, "entities": [{"start": 0, "end": 2, "type": "State"}]}]
        result = score(self.gold, pred)["micro"]
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (0, 1, 2))

    def test_missing_record_is_rejected(self):
        with self.assertRaises(ValueError):
            score(self.gold, [])

    def test_invalid_boundary_is_rejected(self):
        with self.assertRaises(ValueError):
            score(self.gold, [{"record_index": 0, "entities": [{"start": 0, "end": 3, "type": "State"}]}])

    def test_duplicate_prediction_cannot_inflate_score(self):
        with self.assertRaises(ValueError):
            score(self.gold, [{"record_index": 0, "entities": [self.gold[0]["entities"][0]] * 2}])


if __name__ == "__main__":
    unittest.main()
