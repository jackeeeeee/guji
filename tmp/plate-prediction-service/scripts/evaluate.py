"""Recompute every number in report.md from a results CSV.

Usage: ./venv/bin/python scripts/evaluate.py results/hyperlpr3_results.csv
Prints the accuracy table; also writes results/summary.json.
No number is hard-coded: everything is counted from the CSV.
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path


def main():
    path = Path(sys.argv[1])
    rows = list(csv.DictReader(open(path, encoding='utf-8')))
    out = {}
    for kind in ('green', 'blue'):
        sub = [r for r in rows if r['type'] == kind]
        if not sub:
            continue
        full = sum(int(r['full_correct']) for r in sub)
        cc = sum(int(r['char_correct']) for r in sub)
        ct = sum(int(r['char_total']) for r in sub)
        detected = sum(1 for r in sub if r['predicted'])
        times = [float(r['time_sec']) for r in sub]
        d = {
            'n': len(sub),
            'detected': detected,
            'detected_rate': detected / len(sub) if sub else 0,
            'full_correct': full,
            'full_accuracy': full / len(sub) if sub else 0,
            'char_correct': cc, 'char_total': ct,
            'char_accuracy': cc / ct if ct else 0,
            'mean_time_sec': sum(times) / len(times) if times else 0,
            'total_time_sec': sum(times),
        }
        if kind == 'green':
            df_rows = [r for r in sub if r['df_char_correct'] != '']
            df_ok = sum(1 for r in df_rows if r['df_char_correct'] == 'True')
            d['df_n'] = len(df_rows)
            d['df_correct'] = df_ok
            d['df_accuracy'] = df_ok / len(df_rows) if df_rows else 0
            # Error taxonomy (counted, not invented)
            errs = Counter()
            for r in sub:
                if r['full_correct'] == '1':
                    continue
                gt, pred = r['ground_truth'], r['predicted']
                if not pred:
                    errs['no_detection'] += 1
                elif len(pred) != len(gt):
                    errs['length_mismatch'] += 1
                elif len(pred) >= 3 and pred[2] != gt[2]:
                    errs['df_char_wrong'] += 1
                else:
                    errs['other_char_wrong'] += 1
            d['error_taxonomy'] = dict(errs)
        out[kind] = d
    out['total_rows'] = len(rows)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    Path('results/summary.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
