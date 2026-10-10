"""Validate completed pilot artifacts and publish a compact, auditable report."""
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import psutil

from run_pilot import read_rows
from score_spans import score


def main():
    protocol_path = Path('configs/pilot.json')
    protocol = json.loads(protocol_path.read_text())
    expected_protocol = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    expected_split = hashlib.sha256(Path('data/processed/split_manifest.json').read_bytes()).hexdigest()
    expected_source = hashlib.sha256(Path('scripts/run_pilot.py').read_bytes()).hexdigest()
    comparison = json.loads(Path('runs/pilot/comparison.json').read_text())
    hardware_path = Path('runs/pilot/hardware.json')
    if hardware_path.exists():
        hardware = json.loads(hardware_path.read_text())
    else:
        chip = (subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
                if platform.system() == 'Darwin' else platform.processor())
        hardware = {'chip': chip, 'physical_memory_bytes': psutil.virtual_memory().total,
                    'device': protocol['device']}
        hardware_path.write_text(json.dumps(hardware, indent=2) + '\n')
    comparison['hardware'] = hardware
    results = comparison['results']
    assert {r['model'] for r in results} == set(protocol['models'])
    validation = read_rows(protocol['validation_file'])
    for result in results:
        assert result['protocol_sha256'] == expected_protocol
        assert result['split_manifest_sha256'] == expected_split
        assert result['source_sha256'] == expected_source
        assert result['status'] == 'completed' and not result['test_evaluated']
        assert result['checkpoint_reload_identical']
        assert len(result['history']) == protocol['epochs']
        predictions = read_rows(Path('runs/pilot') / result['model'] / 'best_validation_predictions.jsonl')
        assert score(validation, predictions) == result['validation']
    best = max(r['validation']['micro']['f1'] for r in results)
    eligible = [r for r in results if best - r['validation']['micro']['f1'] <= .01]
    selected = min(eligible, key=lambda r: r['training_validation_seconds'])['model']
    assert comparison['selected_for_next_stage'] == selected
    out = Path('benchmarks/pilot-v1')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')
    lines = ['# Encoder feasibility pilot results', '',
             'Protocol registered before training: [pilot protocol](pilot_protocol.md).', '',
             'Full training pool, fixed validation split, seed 42, five epochs, float32 MPS eager attention. Test data was not evaluated.', '',
             f"Hardware: {hardware['chip']}, {hardware['physical_memory_bytes']/2**30:g} GiB physical memory, {hardware['device']}.", '',
             '| Model | Best validation micro-F1 | Best epoch | Training + validation (min) | Sampled RSS (GiB) | Sampled MPS driver (GiB) |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in sorted(results, key=lambda r: list(protocol['models']).index(r['model'])):
        lines.append(f"| {r['model']} | {r['validation']['micro']['f1']:.4f} | {r['best_epoch']} | {r['training_validation_seconds']/60:.2f} | {r['sampled_max_rss_bytes']/2**30:.2f} | {r['sampled_max_mps_driver_bytes']/2**30:.2f} |")
    lines += ['', f'Working model selected by the preregistered rule: **{selected}**.', '',
              'Within 0.01 absolute F1 of the highest score, prefer shorter summed training and validation time. All candidates completed and checkpoint reload reproduced their saved validation predictions.', '',
              '## Per-type validation F1', '', '| Model | Activity | PhysicalObject | Process | Property | State |', '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in results:
        values = [r['validation']['per_type'][label]['f1'] for label in ['Activity', 'PhysicalObject', 'Process', 'Property', 'State']]
        lines.append('| ' + r['model'] + ' | ' + ' | '.join(f'{v:.4f}' for v in values) + ' |')
    lines += ['', '## Interpretation and limits', '',
              '- Single-seed, short-budget selection; not proof of statistical superiority or final five-seed active-learning results.',
              '- Fixed common hyperparameters and five epochs may not reveal each candidate\'s best achievable performance.',
              '- Validation model/epoch selection makes these scores unsuitable as an unbiased final performance estimate.',
              '- Property has only four validation mentions; its score is particularly unstable.',
              '- Timing excludes download/loading, checkpoint I/O, and reload verification; it includes first-step device kernel setup.',
              '- Sampled memory is not an exact peak. RSS and MPS driver allocations share unified memory; do not add them.',
              '- MPS/eager speed comparisons may differ on CUDA or with FlashAttention. Inherent pretrained/tokenizer differences remain.',
              '- Next: keep the chosen working model fixed, define acquisition protocol, and run the repeated sampling benchmark.', '',
              'Machine-readable metrics, epoch history, software versions, and protocol/source/split checksums: [comparison.json](../benchmarks/pilot-v1/comparison.json).', '']
    Path('docs/pilot_results.md').write_text('\n'.join(lines))
    print('\n'.join(lines[:15]))


if __name__ == '__main__':
    main()
