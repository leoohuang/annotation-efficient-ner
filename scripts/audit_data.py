"""Audit the original MaintIE gold corpus before choosing data splits."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

GROUPS = {"PhysicalObject", "State", "Process", "Activity", "Property"}


def audit(path):
    raw = path.read_bytes()
    records = json.loads(raw)
    counts = collections.Counter()
    texts = collections.defaultdict(list)
    errors, overlaps = [], []
    token_count = 0
    bio_records = []
    for i, record in enumerate(records):
        tokens = record["tokens"]
        token_count += len(tokens)
        texts[record["text"]].append(i)
        tags = ["O"] * len(tokens)
        for entity in record["entities"]:
            start, end = entity["start"], entity["end"]
            group = entity["type"].split("/")[0]
            if group not in GROUPS or not 0 <= start < end <= len(tokens):
                errors.append({"record_index": i, "entity": entity})
                continue
            counts[group] += 1
            if any(tag != "O" for tag in tags[start:end]):
                overlaps.append({"record_index": i, "entity": entity})
            tags[start] = "B-" + group
            tags[start + 1:end] = ["I-" + group] * (end - start - 1)
        bio_records.append({"record_index": i, "tokens": tokens, "ner_tags": tags})
    report = {
        "source": "https://github.com/nlp-tlp/maintie/blob/main/data/gold_release.json",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "records": len(records),
        "original_tokens": token_count,
        "entity_mentions": sum(counts.values()),
        "top_level_entity_counts": dict(sorted(counts.items())),
        "duplicate_text_groups": [ids for ids in texts.values() if len(ids) > 1],
        "invalid_entities": errors,
        "overlapping_entities": overlaps,
        "records_with_overlaps": len({row["record_index"] for row in overlaps}),
        "notes": [
            "Indices are zero-based; entity end offsets are exclusive.",
            "Budget counts original corpus tokens, not transformer subtokens.",
            "No training/validation/test split has been chosen.",
            "Group duplicate texts when creating splits; inspect near duplicates separately.",
            "Gold labels will only be used as simulated annotation, never as acquisition features.",
        ],
    }
    return report, bio_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/raw/gold_release.json"))
    parser.add_argument("--out", type=Path, default=Path("reports"))
    args = parser.parse_args()
    report, bio_records = audit(args.data)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "data_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    # A flat BIO sequence cannot faithfully represent overlapping annotations.
    if report["invalid_entities"]:
        raise SystemExit("Invalid entities; inspect reports/data_audit.json.")
    if report["overlapping_entities"]:
        print("Audit complete. BIO conversion withheld: nested/overlapping entities require a documented policy.")
        print(json.dumps({k: v for k, v in report.items() if k != "overlapping_entities"}, indent=2))
        return
    processed = args.out.parent / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    (processed / "gold_top_level_bio.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in bio_records)
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
