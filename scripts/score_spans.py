"""Exact boundary-and-type scoring, including nested entities (no BIO conversion)."""
import argparse
import json
from pathlib import Path

from audit_data import GROUPS


def entity_set(rows, gold_lengths=None):
    values, record_ids = set(), set()
    for row in rows:
        index = row["record_index"]
        if index in record_ids:
            raise ValueError(f"Repeated record_index: {index}")
        record_ids.add(index)
        if gold_lengths is not None:
            if index not in gold_lengths:
                raise ValueError(f"Unknown prediction record: {index}")
            length = gold_lengths[index]
        else:
            length = len(row["tokens"])
        for e in row["entities"]:
            if not (isinstance(e["start"], int) and isinstance(e["end"], int)
                    and 0 <= e["start"] < e["end"] <= length and e["type"] in GROUPS):
                raise ValueError(f"Invalid entity in record {index}: {e}")
            key = (index, e["start"], e["end"], e["type"])
            if key in values:
                raise ValueError(f"Duplicate entity: {key}")
            values.add(key)
    return values, record_ids


def counts(gold, predicted):
    tp = len(gold & predicted)
    fp, fn = len(predicted - gold), len(gold - predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return dict(tp=tp, fp=fp, fn=fn, support=len(gold), precision=precision, recall=recall, f1=f1)


def score(gold_rows, prediction_rows):
    gold, gold_ids = entity_set(gold_rows)
    lengths = {row["record_index"]: len(row["tokens"]) for row in gold_rows}
    predicted, predicted_ids = entity_set(prediction_rows, lengths)
    if gold_ids != predicted_ids:
        raise ValueError("Provide one prediction row for every gold record, including empty predictions.")
    per_type = {label: counts({e for e in gold if e[-1] == label},
                              {e for e in predicted if e[-1] == label}) for label in sorted(GROUPS)}
    return {"micro": counts(gold, predicted), "per_type": per_type,
            "macro_f1": sum(row["f1"] for row in per_type.values()) / len(GROUPS),
            "macro_policy": "all five predefined types; undefined class scores are zero"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    args = parser.parse_args()
    read = lambda path: [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    print(json.dumps(score(read(args.gold), read(args.predictions)), indent=2))


if __name__ == "__main__":
    main()
