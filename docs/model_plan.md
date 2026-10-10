# Nested span NER implementation plan

Decision date: 2026-10-10. This records the initial design. Implementation and the subsequent encoder feasibility pilot are now complete; see [pilot results](pilot_results.md).

## Task

Predict all annotated spans at the five top-level MaintIE types, including nested spans. Do not flatten gold annotations or discard overlapping predictions. The raw corpus has no same-span conflicting top-level types and no cross-type overlapping annotations. All 3,397 raw annotations survive the type mapping.

For example, `air filters plugged` can produce `(air filters, PhysicalObject)`, `(filters, PhysicalObject)`, and `(plugged, State)` simultaneously.

## Implemented architecture

Use a pretrained bidirectional encoder with a small span classification head. A span representation combines its first/last word representations, a pooled representation within the span, and a width embedding. Classify each candidate as NONE or one of the five entity types. Keep the head and training protocol identical when comparing the three candidate encoders; change only the encoder/tokenizer.

Enumerate all contiguous spans in each record rather than choosing a maximum span width from held-out gold labels. The current corpus has at most 12 original tokens per record and 22,095 total candidate spans, so full enumeration is feasible in principle. All three candidates subsequently completed the registered MPS pilot; see the result report. Decode every span whose predicted class is not NONE, without a non-overlap suppression rule.

Preserve original tokens and token offsets. Use tokenizer word alignment to pool subwords for each original token. Assert that every token is represented; fail explicitly if inputs are truncated. Special tokens and padding are excluded from span representations and annotation budgets.

## Pilot and protocol boundaries

- First validate tensor shapes, token alignment, gradient updates, save/load, and prediction format on a small training-only subset.
- Then compare BERT-base, RoBERTa-base, and ModernBERT-base under a predefined pilot budget, using validation performance, runtime, and stability. No test-based selection.
- Handle the many NONE spans through a predefined loss/negative-sampling rule selected on training and validation only. Do not change this rule between acquisition strategies.
- Use the same encoder and head for Random, Uncertainty, and Hybrid runs. Retraining policy, learning rate, early stopping, initial labeled subset, and acquisition batch sizes must be specified before main experiments.
- At acquisition time, scores may use tokens and model outputs but not unrevealed gold entities. Span entropy aggregation needs a length-controlled rule to avoid preferring records simply because they contain more candidate spans; this remains to be specified.
- Full-data means the training pool only. Token budget percentages use the current training pool's 4,313 original tokens; validation/test labels are excluded from that denominator. Report their fixed labeling cost separately when discussing total annotation cost.
- The grouped split differs from any published MaintIE split. Do not claim exact reproduction of published benchmark scores.
- Property has only 35 total mentions (22/4/9 across the current splits); rare-type conclusions will have high uncertainty.

## Sources

MaintIE annotation guidelines explicitly include nested entities (§2.3.6):
https://github.com/nlp-tlp/maintie/blob/main/GUIDELINES.md

MaintIE's model documentation includes SpERT, supporting a span-based approach as a relevant baseline family:
https://github.com/nlp-tlp/maintie/blob/main/MODELS.md

SpERT reference implementation:
https://github.com/lavis-nlp/spert

This proposed NER-only head is inspired by span classification, not an implementation or reproduction of SpERT's joint entity-and-relation architecture. The study contribution remains controlled annotation-efficiency evaluation, not a new NER architecture.
