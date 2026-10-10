import json, random, hashlib, collections
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "gold_release.json"
PROC = ROOT / "data" / "processed"
SPLITS = ROOT / "data" / "splits"
PROC.mkdir(parents=True, exist_ok=True); SPLITS.mkdir(parents=True, exist_ok=True)

TOP = ["PhysicalObject", "State", "Process", "Activity", "Property"]
RATIOS = (0.70, 0.10, 0.20)          # 跑任何实验之前定好，之后不改
START_SEED = 13
MIN_PROPERTY = {"train_pool": 15, "val": 2, "test": 5}
BUDGET_GRID = [0.05, 0.10, 0.20, 0.40, 0.60]


def spans_to_bio(tokens, entities):
    tags = ["O"] * len(tokens)
    n_overlap = 0
    for e in sorted(entities, key=lambda x: (x["start"], -(x["end"] - x["start"]))):
        s, t, typ = e["start"], e["end"], e["type"].split("/")[0]
        if any(tag != "O" for tag in tags[s:t]):
            n_overlap += 1           # 嵌套的内层实体：丢弃，保留最外层
            continue
        tags[s] = f"B-{typ}"
        for i in range(s + 1, t):
            tags[i] = f"I-{typ}"
    return tags, n_overlap


def norm(rec):
    return " ".join(rec["tokens"]).lower()


def split_once(records, seed):
    groups = collections.defaultdict(list)
    for i, r in enumerate(records):
        groups[norm(r)].append(i)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n = len(records)
    cut1, cut2 = RATIOS[0] * n, (RATIOS[0] + RATIOS[1]) * n
    split = {"train_pool": [], "val": [], "test": []}
    seen = 0
    for k in keys:
        name = "train_pool" if seen < cut1 else "val" if seen < cut2 else "test"
        split[name] += groups[k]
        seen += len(groups[k])
    return {k: sorted(v) for k, v in split.items()}


def stats_of(records, split):
    out = {}
    for name, ids in split.items():
        ents = collections.Counter()
        for i in ids:
            for tag in records[i]["tags"]:
                if tag.startswith("B-"):
                    ents[tag[2:]] += 1
        out[name] = {
            "n_records": len(ids),
            "n_tokens": sum(len(records[i]["tokens"]) for i in ids),
            "entities": {t: ents[t] for t in TOP},
        }
    return out


# 1. 读取并转成 BIO
raw = json.load(open(RAW, encoding="utf-8"))
records, total_overlap = [], 0
for i, d in enumerate(raw):
    tags, n_ov = spans_to_bio(d["tokens"], d["entities"])
    total_overlap += n_ov
    records.append({"id": i, "tokens": d["tokens"], "tags": tags})
print("被丢弃的嵌套实体总数:", total_overlap, "（预期 362）")

with open(PROC / "gold_bio.jsonl", "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# 2. 划分，直到满足 Property 的最低数量要求
seed, tries = START_SEED, 1
while True:
    split = split_once(records, seed)
    st = stats_of(records, split)
    if all(st[n]["entities"]["Property"] >= MIN_PROPERTY[n] for n in split):
        break
    seed += 1; tries += 1
    if tries > 1000:
        raise SystemExit("找不到满足条件的种子，请放宽 MIN_PROPERTY")

# 3. 检查无重叠，保存
assert not (set(split["train_pool"]) & set(split["val"]))
assert not (set(split["train_pool"]) & set(split["test"]))
assert not (set(split["val"]) & set(split["test"]))
assert sum(len(v) for v in split.values()) == len(records)

split_hash = hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest()[:16]
pool_tokens = st["train_pool"]["n_tokens"]
out = {
    "seed": seed, "tries": tries, "ratios": RATIOS, "split_hash": split_hash,
    "train_pool": split["train_pool"], "val": split["val"], "test": split["test"],
    "stats": st,
    "budget_tokens": {str(f): round(f * pool_tokens) for f in BUDGET_GRID},
}
json.dump(out, open(SPLITS / "split_v1.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

print("用的种子:", seed, "| 尝试次数:", tries, "| split_hash:", split_hash)
for name in ("train_pool", "val", "test"):
    s = st[name]
    print(f"{name:11s} 记录 {s['n_records']:4d} | token {s['n_tokens']:5d} | {s['entities']}")
print("预算点对应的 token 数:", out["budget_tokens"])
