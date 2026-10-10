"""Run the preregistered encoder feasibility pilot without opening the test split."""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

os.environ.setdefault('HF_HOME', str(Path('.cache/huggingface').resolve()))
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import psutil
import torch
from torch import nn
import torch.nn.functional as F
import transformers
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from score_spans import score

LABELS = ['NONE', 'Activity', 'PhysicalObject', 'Process', 'Property', 'State']


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def encode_rows(rows, tokenizer, maximum):
    encoded = []
    for row in rows:
        feature = tokenizer(row['tokens'], is_split_into_words=True, truncation=False,
                            return_attention_mask=True)
        word_ids = feature.word_ids()
        if len(feature['input_ids']) > maximum:
            raise ValueError(f"Record {row['record_index']} exceeds fixed subtoken limit; no silent truncation")
        if set(x for x in word_ids if x is not None) != set(range(len(row['tokens']))):
            raise ValueError(f"Incomplete original-token alignment: {row['record_index']}")
        gold = {(e['start'], e['end']): LABELS.index(e['type']) for e in row['entities']}
        if len(gold) != len(row['entities']):
            raise ValueError('Multiple types for one span require multilabel classification')
        spans = [(start, end) for start in range(len(row['tokens']))
                 for end in range(start + 1, len(row['tokens']) + 1)]
        encoded.append({'row': row, 'feature': dict(feature), 'word_ids': word_ids,
                        'spans': spans, 'targets': [gold.get(span, 0) for span in spans]})
    return encoded


def collate(items, tokenizer, device):
    inputs = tokenizer.pad([item['feature'] for item in items], return_tensors='pt')
    word_count = max(len(item['row']['tokens']) for item in items)
    pooling = torch.zeros(len(items), word_count, inputs['input_ids'].shape[1])
    coordinates, targets = [], []
    for batch_index, item in enumerate(items):
        for subtoken, word in enumerate(item['word_ids']):
            if word is not None:
                pooling[batch_index, word, subtoken] = 1
        for (start, end), target in zip(item['spans'], item['targets']):
            coordinates.append((batch_index, start, end))
            targets.append(target)
    pooling = pooling / pooling.sum(-1, keepdim=True).clamp_min(1)
    return ({key: value.to(device) for key, value in inputs.items()}, pooling.to(device),
            torch.tensor(coordinates, dtype=torch.long, device=device),
            torch.tensor(targets, dtype=torch.long, device=device))


class SpanNER(nn.Module):
    def __init__(self, encoder, protocol):
        super().__init__()
        self.encoder = encoder
        self.width = nn.Embedding(protocol['max_subtokens'] + 1, protocol['span_width_embedding'])
        hidden = encoder.config.hidden_size
        self.head = nn.Sequential(nn.Dropout(protocol['dropout']),
            nn.Linear(3 * hidden + protocol['span_width_embedding'], protocol['head_hidden_size']),
            nn.GELU(), nn.Dropout(protocol['dropout']),
            nn.Linear(protocol['head_hidden_size'], len(LABELS)))

    def forward(self, inputs, pooling, coordinates):
        hidden = self.encoder(**inputs).last_hidden_state
        words = torch.bmm(pooling, hidden)
        prefix = torch.cat([torch.zeros_like(words[:, :1]), words.cumsum(1)], dim=1)
        batch, start, end = coordinates.unbind(-1)
        width = end - start
        mean = (prefix[batch, end] - prefix[batch, start]) / width.unsqueeze(-1)
        representation = torch.cat([words[batch, start], words[batch, end - 1],
                                    mean, self.width(width)], dim=-1)
        return self.head(representation)


def sync(device):
    if device == 'mps':
        torch.mps.synchronize()
    elif device.startswith('cuda'):
        torch.cuda.synchronize()


def evaluate(model, items, tokenizer, protocol, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for offset in range(0, len(items), protocol['batch_size']):
            batch = items[offset:offset + protocol['batch_size']]
            inputs, pooling, coordinates, _ = collate(batch, tokenizer, device)
            classes = model(inputs, pooling, coordinates).argmax(-1).cpu().tolist()
            position = 0
            for item in batch:
                entities = []
                for start, end in item['spans']:
                    label = classes[position]
                    position += 1
                    if label:
                        entities.append({'start': start, 'end': end, 'type': LABELS[label]})
                predictions.append({'record_index': item['row']['record_index'], 'entities': entities})
    return score([item['row'] for item in items], predictions), predictions


def run_one(name, protocol, protocol_path):
    device = protocol['device']
    if device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('Protocol requires MPS; choose a new registered device before running')
    random.seed(protocol['seed'])
    torch.manual_seed(protocol['seed'])
    torch.set_num_threads(4)
    if device == 'mps':
        torch.mps.manual_seed(protocol['seed'])
    out = Path('runs/pilot') / name
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'result.json').exists():
        raise FileExistsError(f'Existing result at {out}; use a fresh run rather than overwriting evidence')
    model_info = protocol['models'][name]
    started = time.perf_counter()
    tokenizer_kwargs = {'add_prefix_space': True} if name == 'roberta' else {}
    tokenizer = AutoTokenizer.from_pretrained(model_info['id'], revision=model_info['revision'],
                                              use_fast=True, **tokenizer_kwargs)
    config = AutoConfig.from_pretrained(model_info['id'], revision=model_info['revision'])
    if name == 'modernbert':
        config.reference_compile = False
    model_kwargs = {'add_pooling_layer': False} if name in {'bert', 'roberta'} else {}
    encoder = AutoModel.from_pretrained(model_info['id'], revision=model_info['revision'],
        config=config, attn_implementation=protocol['attention'], **model_kwargs)
    # The same hidden dimension and seeded head initialization for all three candidates.
    torch.manual_seed(protocol['seed'])
    if device == 'mps':
        torch.mps.manual_seed(protocol['seed'])
    model = SpanNER(encoder, protocol).to(device)
    train = encode_rows(read_rows(protocol['train_file']), tokenizer, protocol['max_subtokens'])
    validation = encode_rows(read_rows(protocol['validation_file']), tokenizer, protocol['max_subtokens'])
    load_seconds = time.perf_counter() - started
    params = [{'params': list(model.encoder.parameters()), 'lr': protocol['encoder_lr']},
              {'params': list(model.width.parameters()) + list(model.head.parameters()), 'lr': protocol['head_lr']}]
    optimizer = torch.optim.AdamW(params, weight_decay=protocol['weight_decay'])
    steps = math.ceil(len(train) / protocol['batch_size']) * protocol['epochs']
    scheduler = get_linear_schedule_with_warmup(optimizer, int(steps * protocol['warmup_fraction']), steps)
    weights = torch.tensor([protocol['none_loss_weight']] + [1.] * 5, device=device)
    history, best_f1, best_epoch = [], -1., None
    maximum_rss, maximum_mps_allocated, maximum_mps_driver = 0, 0, 0
    process = psutil.Process()

    def sample_memory():
        nonlocal maximum_rss, maximum_mps_allocated, maximum_mps_driver
        maximum_rss = max(maximum_rss, process.memory_info().rss)
        if device == 'mps':
            maximum_mps_allocated = max(maximum_mps_allocated, torch.mps.current_allocated_memory())
            maximum_mps_driver = max(maximum_mps_driver, torch.mps.driver_allocated_memory())

    print(json.dumps({'model': name, 'stage': 'loaded', 'parameters': sum(p.numel() for p in model.parameters()),
                      'train_records': len(train), 'validation_records': len(validation),
                      'load_seconds': round(load_seconds, 2)}), flush=True)
    for epoch in range(1, protocol['epochs'] + 1):
        model.train()
        order = list(range(len(train)))
        random.Random(protocol['seed'] + epoch).shuffle(order)
        sync(device)
        epoch_started = time.perf_counter()
        losses = []
        for step, offset in enumerate(range(0, len(order), protocol['batch_size'])):
            items = [train[i] for i in order[offset:offset + protocol['batch_size']]]
            inputs, pooling, coordinates, targets = collate(items, tokenizer, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs, pooling, coordinates)
            loss = F.cross_entropy(logits, targets, weight=weights)
            if not torch.isfinite(loss).item():
                raise FloatingPointError('Non-finite training loss')
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), protocol['gradient_clip'])
            if not torch.isfinite(gradient_norm).item():
                raise FloatingPointError('Non-finite gradients')
            sample_memory()
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
            if step % 12 == 0:
                print(json.dumps({'model': name, 'epoch': epoch, 'step': step + 1,
                                  'steps_per_epoch': math.ceil(len(train) / protocol['batch_size']),
                                  'loss': round(losses[-1], 4)}), flush=True)
        sync(device)
        train_seconds = time.perf_counter() - epoch_started
        validation_started = time.perf_counter()
        metrics, predictions = evaluate(model, validation, tokenizer, protocol, device)
        sync(device)
        validation_seconds = time.perf_counter() - validation_started
        sample_memory()
        entry = {'epoch': epoch, 'mean_batch_loss': sum(losses) / len(losses),
                 'train_seconds': train_seconds, 'validation_seconds': validation_seconds,
                 'validation': metrics}
        history.append(entry)
        (out / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
        print(json.dumps({'model': name, 'epoch': epoch, 'validation_f1': metrics['micro']['f1'],
                          'train_seconds': round(train_seconds, 2)}), flush=True)
        if metrics['micro']['f1'] > best_f1:
            best_f1, best_epoch = metrics['micro']['f1'], epoch
            torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, out / 'best.pt')
            (out / 'best_validation_predictions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in predictions))
    # Reloading into the same architecture validates checkpoint restoration and repeat decoding.
    optimizer.zero_grad(set_to_none=True)
    del optimizer, scheduler
    gc.collect()
    model.load_state_dict(torch.load(out / 'best.pt', map_location='cpu', weights_only=True))
    reloaded_metrics, reloaded_predictions = evaluate(model, validation, tokenizer, protocol, device)
    saved = read_rows(out / 'best_validation_predictions.jsonl')
    if saved != reloaded_predictions or reloaded_metrics['micro']['f1'] != best_f1:
        raise AssertionError('Checkpoint reload changed validation predictions')
    result = {'status': 'completed', 'model': name, 'model_info': model_info, 'seed': protocol['seed'],
              'best_epoch': best_epoch, 'validation': reloaded_metrics, 'history': history,
              'training_validation_seconds': sum(e['train_seconds'] + e['validation_seconds'] for e in history),
              'load_seconds': load_seconds, 'sampled_max_rss_bytes': maximum_rss,
              'sampled_max_mps_allocated_bytes': maximum_mps_allocated,
              'sampled_max_mps_driver_bytes': maximum_mps_driver,
              'checkpoint_reload_identical': True, 'test_evaluated': False,
              'parameter_count': sum(p.numel() for p in model.parameters()),
              'protocol_sha256': hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
              'split_manifest_sha256': hashlib.sha256(Path('data/processed/split_manifest.json').read_bytes()).hexdigest(),
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                  'transformers': transformers.__version__, 'platform': platform.platform(),
                  'device': device, 'mps_fallback_enabled': os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1'},
              'limitations': ['One seed, short fixed training budget; not a statistical superiority claim.',
                  'Memory is sampled, not an exact peak; RSS and MPS driver share unified memory and must not be summed.',
                  'MPS results are hardware-specific and may not be bitwise reproducible.']}
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'model': name, 'status': 'completed', 'best_epoch': best_epoch,
                      'validation_f1': best_f1}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, default=Path('configs/pilot.json'))
    parser.add_argument('--model', choices=['bert', 'roberta', 'modernbert'])
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if args.model:
        run_one(args.model, protocol, args.protocol)
    else:
        # Separate processes prevent allocator state from contaminating model memory comparisons.
        for name in protocol['models']:
            completed = subprocess.run([sys.executable, __file__, '--protocol', str(args.protocol), '--model', name])
            if completed.returncode:
                print(json.dumps({'model': name, 'status': 'failed', 'exit_code': completed.returncode}), flush=True)
        results = [json.loads(path.read_text()) for path in Path('runs/pilot').glob('*/result.json')]
        if len(results) != len(protocol['models']):
            raise SystemExit('Not all candidates completed. Inspect failures before making a selection.')
        best = max(r['validation']['micro']['f1'] for r in results)
        eligible = [r for r in results if best - r['validation']['micro']['f1'] <= .01]
        selected = min(eligible, key=lambda r: r['training_validation_seconds'])['model']
        comparison = {'selected_for_next_stage': selected, 'selection_rule': protocol['selection_rule'],
                      'results': results, 'status': 'single-seed feasibility selection', 'test_evaluated': False}
        Path('runs/pilot/comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')
        print(json.dumps({'selected_for_next_stage': selected}), flush=True)


if __name__ == '__main__':
    main()
