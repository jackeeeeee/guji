"""Run HyperLPR3 (ONNX Runtime, CPU) over a sampled CCPD set.

Usage:
  ./venv/bin/python scripts/run_hyperlpr3.py \
      --green-dir data/ccpd2020/ccpd_green/test \
      --blue-dir  data/ccpd2019/ccpd_base \
      --n-green 300 --n-blue 100 --seed 42 \
      --out results/hyperlpr3_results.csv

Sampling: deterministic (sorted filenames, fixed seed shuffle) so the
exact sample can be reproduced from the CSV + seed. Ground truth is
parsed from CCPD filenames only (see scripts/ccpd_parse.py).
Prediction: HyperLPR3 LicensePlateCatcher, first plate returned;
empty prediction is recorded as '' with detected=False (not imputed).
"""
import argparse
import csv
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ccpd_parse import parse_ccpd_filename

import cv2
import numpy as np


def char_accuracy(gt, pred):
    """Per-character accuracy over aligned positions (min length)."""
    if not pred:
        return 0, len(gt)
    n = min(len(gt), len(pred))
    correct = sum(1 for a, b in zip(gt[:n], pred[:n]) if a == b)
    return correct, len(gt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--green-dir', required=True)
    ap.add_argument('--blue-dir', default='')
    ap.add_argument('--n-green', type=int, default=300)
    ap.add_argument('--n-blue', type=int, default=100)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    from hyperlpr3 import LicensePlateCatcher
    catcher = LicensePlateCatcher()
    print('HyperLPR3 catcher ready', flush=True)

    jobs = []
    for kind, d, n in (('green', args.green_dir, args.n_green),
                       ('blue', args.blue_dir, args.n_blue)):
        if not d or n <= 0:
            continue
        files = sorted(Path(d).glob('*.jpg'))
        rng = random.Random(args.seed)
        rng.shuffle(files)
        chosen = files[:n]
        print(f'{kind}: dir={d} total={len(files)} sampled={len(chosen)}', flush=True)
        for f in chosen:
            jobs.append((kind, f))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    t_all = time.time()
    for i, (kind, f) in enumerate(jobs, 1):
        try:
            gt, _idx, _verts, bright, blur = parse_ccpd_filename(f.name)
        except Exception as e:
            print(f'SKIP {f.name}: {e}', flush=True)
            continue
        img = cv2.imread(str(f))
        if img is None:
            print(f'SKIP unreadable {f}', flush=True)
            continue
        t0 = time.time()
        try:
            plates = catcher(img)
        except Exception as e:
            plates = []
            print(f'ERROR {f.name}: {e}', flush=True)
        dt = time.time() - t0
        # HyperLPR3 0.1.3 multitask pipeline returns plate.to_result():
        # [plate_code, rec_confidence, plate_type, det_bound_box]; older
        # pipeline returns Plate dicts/objects. Handle all three shapes.
        if plates:
            p = plates[0]
            if isinstance(p, (list, tuple)):
                pred = p[0] or ''
                conf = float(p[1]) if p[1] is not None else ''
                ptype = int(p[2]) if p[2] is not None else ''
            elif isinstance(p, dict):
                pred = p.get('plate_code', '') or ''
                conf = p.get('rec_confidence', '')
                ptype = p.get('plate_type', '')
            else:
                pred = getattr(p, 'plate_code', '') or ''
                conf = getattr(p, 'rec_confidence', '')
                ptype = getattr(p, 'plate_type', '')
        else:
            pred, conf, ptype = '', '', ''
        cc, denom = char_accuracy(gt, pred)
        df_correct = ''
        if kind == 'green' and len(gt) >= 3:
            # NEV 3rd char (index 2) should be D (pure EV) or F (PHEV).
            df_correct = (len(pred) >= 3 and pred[2] == gt[2])
        rows.append({
            'filename': f.name, 'source_dir': str(f.parent), 'type': kind,
            'ground_truth': gt, 'predicted': pred,
            'full_correct': int(pred == gt),
            'pred_len': len(pred), 'gt_len': len(gt),
            'char_correct': cc, 'char_total': denom,
            'df_char_correct': df_correct,
            'confidence': conf, 'plate_type_code': ptype,
            'brightness': bright, 'blur': blur,
            'time_sec': round(dt, 3), 'model': 'hyperlpr3-0.1.3-onnx-cpu',
        })
        if i % 50 == 0 or i == len(jobs):
            print(f'[{i}/{len(jobs)}] {kind} {f.name} gt={gt} pred={pred} '
                  f'correct={pred == gt} {dt:.2f}s', flush=True)
        # Flush incrementally so a crash still leaves partial results.
        with open(out_path, 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f'DONE rows={len(rows)} total_time={time.time() - t_all:.1f}s -> {out_path}')


if __name__ == '__main__':
    main()
