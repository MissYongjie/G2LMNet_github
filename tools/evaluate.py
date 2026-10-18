"""Evaluate change maps against the reference maps (OA, Kappa, F1 with Otsu thresholding).

Works on the released maps (results/) and on training outputs (work_dirs/), which share the
layout <pred_dir>/hetdata<N>/change_map.png.

Example:
    python tools/evaluate.py --pred_dir results --data_dir data/hetdata
    python tools/evaluate.py --pred_dir work_dirs --data_dir data/hetdata --save_vis
"""

import argparse
import csv
import os
import sys

import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from g2lmnet.data import load_mat, resolve_path  # noqa: E402
from g2lmnet.metrics import error_map, evaluate_gray  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", default="results")
    parser.add_argument("--data_dir", default="data/hetdata")
    parser.add_argument("--map_name", default="change_map.png", help="e.g. graph_prior.png to score the graph prior")
    parser.add_argument("--datasets", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--save_vis", action="store_true", help="write <map>_binary.png and <map>_error.png")
    parser.add_argument("--csv", default=None, help="optional output CSV")
    args = parser.parse_args()

    rows = []
    for n in args.datasets:
        pred_path = os.path.join(args.pred_dir, f"hetdata{n}", args.map_name)
        if not os.path.exists(pred_path):
            print(f"HetData #{n}: {pred_path} not found, skipped")
            continue
        with open(os.path.join(ROOT, "configs", f"hetdata{n}.yaml")) as f:
            cfg = yaml.safe_load(f)
        _, _, ref = load_mat(resolve_path(args.data_dir, cfg["dataset"]["file"]))
        gray = np.array(Image.open(pred_path).convert("L"))
        m, pred = evaluate_gray(ref, gray)
        rows.append({"dataset": f"HetData #{n}", **{k: round(v, 4) for k, v in m.items()}})
        if args.save_vis:
            stem = os.path.splitext(pred_path)[0]
            Image.fromarray(pred).save(stem + "_binary.png")
            Image.fromarray(error_map(ref, pred)).save(stem + "_error.png")

    if not rows:
        return
    print(f"\n{'Dataset':<12} {'OA':>7} {'Kappa':>7} {'F1':>7}   (fractions, as in the paper)")
    for r in rows:
        print(f"{r['dataset']:<12} {r['OA'] / 100:7.3f} {r['Kappa'] / 100:7.3f} {r['F1'] / 100:7.3f}")
    if len(rows) > 1:
        print(f"{'Mean':<12} {np.mean([r['OA'] for r in rows]) / 100:7.3f} "
              f"{np.mean([r['Kappa'] for r in rows]) / 100:7.3f} {np.mean([r['F1'] for r in rows]) / 100:7.3f}")
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
