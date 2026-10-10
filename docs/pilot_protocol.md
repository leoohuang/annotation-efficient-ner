# Encoder feasibility pilot protocol

Registered before the first training run on 2026-10-10. Machine-readable settings and pinned model revisions are in [configs/pilot.json](../configs/pilot.json).

- Candidates: google-bert/bert-base-uncased, FacebookAI/roberta-base, answerdotai/ModernBERT-base. The BERT checkpoint is uncased; tokenizer/pretraining differences are inherent to the candidates.
- Fixed training pool: 755 records, 4,313 original tokens. Fixed validation: 155 records. The test corpus is never opened by the pilot.
- One seed (42), five epochs, batch size 16, full encoder fine-tuning, float32, MPS, eager attention. ModernBERT reference compilation is disabled for device compatibility; no CPU operator fallback is enabled.
- The shared head concatenates first word, last word, mean of words within the span, and a 16-dimensional width embedding, then uses a 256-unit GELU MLP with dropout 0.1 and a six-class output (NONE plus five entity types). All original-word spans are enumerated; all nested predictions are retained.
- Subword states are mean-pooled to original words using tokenizer alignment. Padding and special tokens are excluded. Inputs beyond 64 subtokens fail rather than truncate gold entities.
- AdamW: encoder learning rate 2e-5; head/width learning rate 1e-3; weight decay 0.01; 10% warmup and linear decay; gradient norm clipped to 1.0. Weighted cross-entropy over all batch spans: NONE weight 0.2, each entity class weight 1.0. No candidate-specific tuning.
- Each candidate gets the same deterministic record order per epoch. Record best validation entity micro-F1, per-type scores, epoch times, and sampled RSS/MPS memory. Download/loading and checkpoint I/O are excluded from training+validation timing. Separate processes avoid allocator state contamination between candidates.
- Keep the checkpoint with highest validation micro-F1; exact ties keep the earliest epoch. Verify that reloading reproduces its validation predictions.
- Selection: among successful finite runs, take the highest best validation micro-F1. Models within 0.01 absolute F1 are treated as tied; choose the one with shortest summed training+validation time. All three must complete before automatic selection. Inspect any technical failure rather than treating it as proof of inferior modeling quality.

## Limits

This is a single-seed, short-budget feasibility selection, not a five-seed benchmark or a statistical superiority test. Absolute runtime depends on this Apple hardware and eager attention; it does not establish CUDA speed rankings. Memory samples are not exact peaks, and RSS/MPS allocations share unified memory and cannot be added. MPS kernels may not be bitwise deterministic. Validation has only four Property mentions.

## Run

Install the training dependencies for the target device, then from the repository root:

```sh
python scripts/run_pilot.py
```

The local run uses PyTorch 2.14.0, Transformers 4.57.1, and psutil 7.1.0. A different device or software version requires an explicitly recorded protocol change. Results and checkpoints go to ignored `runs/pilot/`; the runner refuses to overwrite an existing candidate result.

Official checkpoint and implementation references:

- https://huggingface.co/google-bert/bert-base-uncased
- https://huggingface.co/FacebookAI/roberta-base
- https://huggingface.co/answerdotai/ModernBERT-base
- https://huggingface.co/docs/transformers/model_doc/modernbert
