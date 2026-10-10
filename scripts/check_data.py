import json, collections
from src.config import RAW_DIR
 
data = json.load(open(RAW_DIR / "gold_release.json", encoding="utf-8"))
n_tok = [len(d["tokens"]) for d in data]
print("texts:", len(data), "| tokens total/avg/max:",
      sum(n_tok), round(sum(n_tok) / len(n_tok), 2), max(n_tok))
 
bad_span = overlap = 0
top, leaf = collections.Counter(), collections.Counter()
for d in data:
    n = len(d["tokens"]); used = [0] * n
    for e in d["entities"]:
        s, t = e["start"], e["end"]
        if not (0 <= s < t <= n):
            bad_span += 1; continue
        if any(used[s:t]):
            overlap += 1
        for i in range(s, t):
            used[i] = 1
        top[e["type"].split("/")[0]] += 1
        leaf[e["type"]] += 1
 
print("bad_span:", bad_span, "| overlap_span:", overlap)
print("top_entities:", top.most_common())
print("leaf", len(leaf))
 
uniq = collections.Counter(" ".join(d["tokens"]).lower() for d in data)
print("唯一文本:", len(uniq), "| 重复率: %.1f%%" % (100 * (1 - len(uniq) / len(data))))
