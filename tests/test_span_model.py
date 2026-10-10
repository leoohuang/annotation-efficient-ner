"""Offline checks for span enumeration, subword pooling, gradients, and checkpoint roundtrip."""
import importlib.util
import io
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ['torch', 'transformers', 'psutil'])
if AVAILABLE:
    import torch
    from torch import nn
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    from run_pilot import SpanNER, collate, encode_rows


@unittest.skipUnless(AVAILABLE, 'Training dependencies are optional for the data-only pipeline')
class SpanModelTests(unittest.TestCase):
    def setUp(self):
        backend = Tokenizer(models.WordPiece({'[UNK]': 0, '[PAD]': 1, 'air': 2, 'filter': 3, '##s': 4}))
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]', pad_token='[PAD]')
        self.rows = [{'record_index': 0, 'tokens': ['air', 'filters'], 'entities': [
            {'start': 0, 'end': 2, 'type': 'PhysicalObject'},
            {'start': 1, 'end': 2, 'type': 'PhysicalObject'}]}]
        self.protocol = {'max_subtokens': 64, 'span_width_embedding': 4, 'dropout': 0., 'head_hidden_size': 8}

    def test_subwords_pool_to_original_tokens_and_nested_spans_survive(self):
        items = encode_rows(self.rows, self.tokenizer, 64)
        inputs, pooling, coordinates, targets = collate(items, self.tokenizer, 'cpu')
        self.assertEqual(inputs['input_ids'].tolist(), [[2, 3, 4]])
        self.assertEqual(pooling.tolist(), [[[1., 0., 0.], [0., .5, .5]]])
        self.assertEqual(coordinates.tolist(), [[0, 0, 1], [0, 0, 2], [0, 1, 2]])
        self.assertEqual(targets.tolist(), [0, 2, 2])

    def test_truncation_is_rejected(self):
        with self.assertRaises(ValueError):
            encode_rows(self.rows, self.tokenizer, 2)

    def test_gradients_and_checkpoint_roundtrip(self):
        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(hidden_size=8)
                self.embedding = nn.Embedding(5, 8)

            def forward(self, input_ids, **kwargs):
                return SimpleNamespace(last_hidden_state=self.embedding(input_ids))

        items = encode_rows(self.rows, self.tokenizer, 64)
        inputs, pooling, coordinates, targets = collate(items, self.tokenizer, 'cpu')
        model = SpanNER(Encoder(), self.protocol)
        logits = model(inputs, pooling, coordinates)
        self.assertEqual(tuple(logits.shape), (3, 6))
        nn.functional.cross_entropy(logits, targets).backward()
        self.assertGreater(model.encoder.embedding.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.head[1].weight.grad.abs().sum().item(), 0)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = SpanNER(Encoder(), self.protocol)
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        torch.testing.assert_close(model(inputs, pooling, coordinates), restored(inputs, pooling, coordinates))


if __name__ == '__main__':
    unittest.main()
