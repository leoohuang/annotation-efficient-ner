"""Preserve nested spans and produce one reproducible, duplicate-grouped split."""
import collections
import difflib
import hashlib
import json
import random
from pathlib import Path

from audit_data import GROUPS, audit

SEED = 20261010


def duplicate_groups(records):
    tokens = [tuple(t.casefold() for t in row["tokens"]) for row in records]
    parent = list(range(len(records)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    pairs = []
    for i, a in enumerate(tokens):
        for j in range(i + 1, len(tokens)):
            b = tokens[j]
            same = a == b
            near = (min(len(a), len(b)) >= 3
                    and min(len(a), len(b)) / max(len(a), len(b)) >= .8
                    and difflib.SequenceMatcher(None, a, b, autojunk=False).ratio() >= .9)
            if same or near:
                parent[root(j)] = root(i)
                pairs.append({"indices": [i, j], "kind": "exact" if same else "near"})
    groups = collections.defaultdict(list)
    for i in range(len(records)):
        groups[root(i)].append(i)
    return list(groups.values()), pairs


def split_indices(records, seed=SEED):
    groups, _ = duplicate_groups(records)
    random.Random(seed).shuffle(groups)
    train_end, validation_end = int(len(groups) * .70), int(len(groups) * .85)
    return {
        "train": sorted(i for group in groups[:train_end] for i in group),
        "validation": sorted(i for group in groups[train_end:validation_end] for i in group),
        "test": sorted(i for group in groups[validation_end:] for i in group),
    }


def to_spans(record, index):
    entities = []
    seen = set()
    for entity in record["entities"]:
        label = entity["type"].split("/")[0]
        key = (entity["start"], entity["end"], label)
        if key in seen:
            raise ValueError(f"Duplicate top-level entity in record {index}: {key}")
        seen.add(key)
        entities.append({"start": key[0], "end": key[1], "type": label})
    return {"record_index": index, "text": record["text"],
            "tokens": record["tokens"], "entities": entities}


def main():
    path = Path("data/raw/gold_release.json")
    report, _ = audit(path)
    if report["invalid_entities"]:
        raise ValueError("Invalid annotations; fix the source issue before preparing data.")
    raw = json.loads(path.read_text())
    records = [to_spans(record, i) for i, record in enumerate(raw)]
    splits = split_indices(raw)
    out = Path("data/processed")
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, indices in splits.items():
        rows = [records[i] for i in indices]
        counts = collections.Counter(e["type"] for row in rows for e in row["entities"])
        if set(counts) != GROUPS:
            raise ValueError(f"Missing entity class in {name}; review split protocol.")
        (out / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        summary[name] = {"records": len(rows), "tokens": sum(len(r["tokens"]) for r in rows),
                         "entities": dict(sorted(counts.items()))}
    groups, pairs = duplicate_groups(raw)
    membership = {i: name for name, ids in splits.items() for i in ids}
    assert all(membership[a] == membership[b] for row in pairs for a, b in [row["indices"]])
    Path("reports").mkdir(exist_ok=True)
    (Path("reports") / "near_duplicates.json").write_text(json.dumps({
        "rule": "casefolded tokens: exact equality or SequenceMatcher >= 0.9, min length 3; connected components",
        "pairs": pairs, "cross_split_pairs": 0, "group_count": len(groups),
    }, indent=2) + "\n")
    manifest = {"seed": SEED, "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "group_rule": "exact and near duplicate connected components; see reports/near_duplicates.json", "group_ratios": [.70, .15, .15],
                "indices": splits, "summary": summary,
                "status": "v1 fixed split, grouped for exact and >=0.9 token similarity",
                "budget_denominator_tokens": summary["train"]["tokens"]}
    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
