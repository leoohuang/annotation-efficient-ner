import json, collections
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "data" / "raw" / "gold_release.json"
data = json.load(open(RAW, encoding="utf-8"))

rel = collections.Counter()
ex = collections.defaultdict(list)
for d in data:
    ents = [(e["start"], e["end"], e["type"]) for e in d["entities"]]
    for i in range(len(ents)):
        for j in range(i + 1, len(ents)):
            (s1, t1, y1), (s2, t2, y2) = ents[i], ents[j]
            if max(s1, s2) >= min(t1, t2):
                continue                                  # 没有重叠
            if (s1, t1) == (s2, t2):
                kind = "完全相同的范围"
            elif (s1 <= s2 and t2 <= t1) or (s2 <= s1 and t1 <= t2):
                kind = "嵌套（一个包含另一个）"
            else:
                kind = "部分交叉"
            same = y1.split("/")[0] == y2.split("/")[0]
            key = (kind, "同顶层类型" if same else "不同顶层类型")
            rel[key] += 1
            if len(ex[key]) < 3:
                ex[key].append((
                    " ".join(d["tokens"]),
                    " ".join(d["tokens"][s1:t1]) + " [" + y1.split("/")[0] + "]",
                    " ".join(d["tokens"][s2:t2]) + " [" + y2.split("/")[0] + "]",
                ))

print("重叠的实体对（按种类）：")
for k, v in rel.most_common():
    print(" ", k, v)
    for e in ex[k]:
        print("     例：", e)

# 「保留最长」规则会丢掉多少实体
dropped, total = collections.Counter(), 0
for d in data:
    ents = sorted(d["entities"], key=lambda e: (-(e["end"] - e["start"]), e["start"]))
    used = [0] * len(d["tokens"])
    for e in ents:
        s, t = e["start"], e["end"]
        total += 1
        if any(used[s:t]):
            dropped[e["type"].split("/")[0]] += 1
            continue
        for k in range(s, t):
            used[k] = 1
print("\n按「保留最长」规则会丢掉:", sum(dropped.values()), "/", total)
print("丢掉的实体按顶层类型：", dict(dropped))
