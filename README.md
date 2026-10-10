# Annotation-efficient NER for industrial maintenance

[中文说明](README.zh-CN.md)

A research project comparing random sampling, uncertainty sampling, and uncertainty plus diversity for named entity recognition in maintenance records, under equal annotation token budgets.

**Status:** data preparation, nested-span training, and the three-encoder feasibility pilot are complete. Active-learning sampling is not implemented yet.

The preregistered single-seed, five-epoch pilot selected **BERT-base-uncased** as the working encoder. Best validation micro-F1: BERT 0.9072, RoBERTa 0.8936, ModernBERT 0.8669. These are validation selection scores, not final test results or evidence of general superiority. See the [pilot report](docs/pilot_results.md), [protocol](docs/pilot_protocol.md), and [machine-readable metrics](benchmarks/pilot-v1/comparison.json).

## Quick start

The current pipeline uses Python 3.12 and the standard library. No GPU or third-party packages are needed for data preparation and tests.

```sh
git clone https://github.com/leoohuang/annotation-efficient-ner.git
cd annotation-efficient-ner
python3 scripts/download_data.py
python3 scripts/audit_data.py
python3 scripts/prepare_data.py
python3 -m unittest discover -s tests -v
```

The download script fetches public MaintIE gold data and its license from a pinned revision, verifies the data SHA-256, and refuses to overwrite different local content. Raw data and generated corpus files are excluded from Git; the split manifest is versioned.

## Dataset and task

[MaintIE's gold corpus](https://github.com/nlp-tlp/maintie) contains 1,076 records, 6,119 original tokens, and 3,397 entity annotations. Fine-grained labels map to PhysicalObject, Activity, State, Process, and Property.

There are nested entities in 328 records. Both `air filters` and `filters`, for example, can be annotated in one text. Conversion preserves every gold entity as a span. The audit script withholds flat BIO conversion when overlaps are detected. The implemented model uses a pretrained encoder and span classification head; see the [model plan](docs/model_plan.md).

## Fixed split

Seed: `20261010`. Case-normalized exact token duplicates and near duplicates (token SequenceMatcher ratio >= 0.9, minimum length 3) form connected groups. Groups are split approximately 70/15/15; record proportions vary slightly.

| Split | Records | Original tokens | Entities |
| --- | ---: | ---: | ---: |
| Training pool | 755 | 4,313 | 2,391 |
| Validation | 155 | 853 | 488 |
| Test | 166 | 953 | 518 |

The [manifest](data/processed/split_manifest.json) records source checksum, seed, indices, and statistics. This is a project-specific split, not a reproduction of published MaintIE splits. Grouping controls the specified surface-similarity cases; equipment/work-order independence cannot be established from the available data. Property has only 4 validation and 9 test mentions, limiting per-class conclusions.

## Evaluation

An entity is correct only when record index, start token, exclusive end token, and type all match. Nested spans are scored independently. Outputs include micro precision/recall/F1, per-type scores, and macro F1 over all five predefined types (undefined scores are zero).

```sh
python3 scripts/score_spans.py --gold data/processed/validation.jsonl --predictions predictions.jsonl
```

Provide one JSON line for every gold record, including empty predictions:

```json
{"record_index": 5, "entities": [{"start": 1, "end": 3, "type": "PhysicalObject"}, {"start": 2, "end": 3, "type": "PhysicalObject"}]}
```

Nine data/scoring tests cover duplicate grouping, nested-span preservation, exact-match scoring, missing records, malformed boundaries, and repeated predictions. Three additional offline training tests check subword pooling, truncation rejection, gradients, and checkpoint restoration; these require the training dependencies. All 12 passed locally. Oracle self-scoring is a pipeline check, not model performance.

## Experiment plan

1. Completed: shared span model, token alignment, gradient updates, and checkpoint restoration.
2. Completed: BERT/RoBERTa/ModernBERT pilot; BERT-base-uncased selected by the registered rule.
3. Fix one model and training protocol; establish full-training-pool and random-sampling curves.
4. Compare uncertainty and uncertainty plus diversity using shared initial labeled sets, equal budgets, and at least five seeds.
5. Report learning curves, per-type results, and token budgets needed to approach full-data F1.

The proposed budget grid is 5/10/20/40/60/100% of training-pool original tokens. Seed size, batch rules, uncertainty aggregation, diversity selection, and retraining policy remain to be specified before main experiments. Validation guides tuning and stopping; test results do not guide these decisions. Simulated labeled-token savings do not directly measure annotator time savings.

## Files

- `scripts/download_data.py`: pinned download and checksum verification.
- `scripts/audit_data.py`: source annotation audit.
- `scripts/prepare_data.py`: nested spans and grouped splits.
- `scripts/score_spans.py`: exact-match entity scoring.
- `scripts/run_pilot.py`: encoder pilot with a shared nested-span classification head.
- `scripts/summarize_pilot.py`: artifact validation and result report.
- `configs/pilot.json`: registered settings and pinned model revisions.
- `tests/`: data, scoring, and optional offline training tests.
- `docs/model_plan.md`: model design and protocol boundaries.
- `docs/pilot_results.md`: measured pilot results and limitations.
- `data/processed/split_manifest.json`: versioned split definition.

## License and attribution

Project code is MIT licensed; see [LICENSE](LICENSE). MaintIE is a separate upstream dataset; its license is downloaded with the data. This repository contains no private company records.

Please cite MaintIE when using the corpus:

Bikaun, T. K., French, T., Stewart, M., Liu, W., and Hodkiewicz, M. (2024). [MaintIE: A Fine-Grained Annotation Schema and Benchmark for Information Extraction from Maintenance Short Texts](https://aclanthology.org/2024.lrec-main.954/). LREC-COLING 2024, 10939–10951.

## Running the training pilot

Use a fresh Python environment and a PyTorch build appropriate for your device. The tested versions are in `requirements-training.txt`. The registered configuration uses Apple MPS; running on another device requires recording a new protocol.

```sh
python3 -m pip install -r requirements-training.txt
python3 scripts/run_pilot.py
python3 scripts/summarize_pilot.py
```

Run the three candidates in separate processes, keeping all settings fixed. Existing candidate result files are protected from overwriting; use a fresh checkout/run for replication. Large checkpoints, model caches, raw data, and prediction files remain local and are excluded from Git. The report generator records hardware metadata in `runs/pilot/hardware.json`.
