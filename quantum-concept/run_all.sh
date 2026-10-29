#!/usr/bin/env bash
# Full reproduction (about 47 min main + 25 min confirmation on 7 CPU workers).
set -euo pipefail
python run_experiments.py --instances 20 --steps 1,2,3 --seed 2026 --workers 7 --out results/probs.npz
python analyze.py --probs results/probs.npz --shot-seeds 20 --tag main
python pooled.py main
python make_figures.py main
python run_experiments.py --instances 10 --steps 1,2,3 --seed 3031 --workers 7 --out results/probs_confirm.npz
python analyze.py --probs results/probs_confirm.npz --shot-seeds 20 --tag confirm
python pooled.py confirm
