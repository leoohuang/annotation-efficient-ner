def spans_to_bio(tokens, entities):
    tags = ["O"] * len(tokens)
    n_overlap = 0
    for e in sorted(entities, key=lambda x: (x["start"], -(x["end"] - x["start"]))):
        s, t, typ = e["start"], e["end"], e["type"].split("/")[0]
        if any(tag != "O" for tag in tags[s:t]):
            n_overlap += 1          # 重叠：保留先到者，并计数写进报告
            continue
        tags[s] = f"B-{typ}"
        for i in range(s + 1, t):
            tags[i] = f"I-{typ}"
    return tags, n_overlap
