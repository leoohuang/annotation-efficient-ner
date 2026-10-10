import torch
print("CUDA:", torch.cuda.is_available(), "| MPS:", torch.backends.mps.is_available())
from transformers import AutoTokenizer
for name in ["bert-base-cased", "roberta-base", "answerdotai/ModernBERT-base"]:
    tok = AutoTokenizer.from_pretrained(name)
    print(name, "OK, vocab =", len(tok))
import seqeval
print("seqeval OK")
