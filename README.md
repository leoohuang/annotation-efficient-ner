# Annotation-Efficient Industrial Maintenance NER

Reproducible data preparation and nested entity evaluation for the public **MaintIE** maintenance corpus, plus a pre-registered encoder feasibility pilot for a span-based NER model.

> Public data only. The active-learning experiments are in [ae-ner-maintenance](https://github.com/leoohuang/ae-ner-maintenance).

## What is here

- **Data pipeline:** download of the pinned MaintIE release with checksum verification, data audit, preprocessing, and a frozen split ([`scripts/`](scripts/), [`data/splits/`](data/splits/)).
- **Nested span model:** Transformer encoder with a span-classification head over all contiguous spans; design in [`docs/model_plan.md`](docs/model_plan.md).
- **Encoder pilot:** BERT, RoBERTa, and ModernBERT compared under a protocol registered before training ([`docs/pilot_protocol.md`](docs/pilot_protocol.md)); results and limits in [`docs/pilot_results.md`](docs/pilot_results.md), machine-readable metrics in [`benchmarks/pilot-v1/`](benchmarks/pilot-v1/).
- **CI:** GitHub Actions runs the unit tests and checks that the frozen split is reproduced exactly ([`.github/workflows/checks.yml`](.github/workflows/checks.yml)).

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # exact versions: requirements.lock
python -m unittest discover -s tests -v
python scripts/download_data.py
python scripts/audit_data.py
python scripts/prepare_data.py
```

## Limitations

- The pilot is a single-seed, short-budget feasibility selection on Apple MPS, not a benchmark or a statistical comparison.
- The split differs from published MaintIE splits; scores are not comparable with published results.

## License

MIT — see [LICENSE](LICENSE).

---

## log 10.10. 
  
overlap:所有 362 个重叠都是同一顶层类型内的嵌套，已合并为最外层实体，占全部实体的 10.7%。