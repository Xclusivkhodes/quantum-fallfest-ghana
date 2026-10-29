# SA-ZNE: Symmetry-Gauged Zero-Noise Extrapolation (a variant/combination)

**One sentence.** Run ordinary noise-amplified ZNE circuits, but instead of trusting the *nominal* noise
factor λ (1, 3, 5), read the *actual* noise strength of every circuit from the fraction of shots that
violate a conserved quantity (Hamming weight), using the very same shots, then extrapolate the
symmetry-post-selected observable against that measured noise axis.

**Status: a variant/combination of known ideas, not a new primitive.** It mixes
symmetry verification (Bonet-Monroig et al. 2018), measured-noise-axis ZNE (IC-ZNE, Koenig et al. 2024,
which uses extra inverted circuits) and symmetry-decay-guided extrapolation (GUESS, arXiv:2603.13060,
which uses an extra impurity circuit). What is different here: the gauge costs **zero extra circuits**,
because it comes from the discarded shots, and it is measured **per circuit execution**, so it
follows calibration drift between the λ circuits.

All numbers below come from simulations actually run in this folder (Qiskit 2.6.1, Aer 0.18.2,
density-matrix simulation with FakeSherbrooke / FakeTorino noise models). No real hardware was used.

## Test problem
- 6-qubit open XXZ chain from the Néel state |010101>, random J∈[0.5,1.5], Δ∈[0.3,1.5], h∈[-1,1], dt=0.25,
  1–3 first-order Trotter steps. Total magnetisation is conserved, so the ideal state lives in the weight-3 sector.
- Observables: staggered magnetisation and mean nearest-neighbour ZZ; the target is the ideal (noise-free) Trotter value.
- Noise amplification: global folding λ = 1, 3, 5 and local 2-qubit-gate folding λ = 1, 2, 3.
- Readout: analytic per-qubit confusion, tensored readout mitigation with a 4000-shot calibration; 4000 multinomial shots per circuit, 20 shot seeds.
- **Drift model:** every λ circuit is run with its gate-error rates multiplied by an independent gain
  g ∈ {0.7, 0.85, 1.0, 1.15, 1.3} (calibration drift between executions). "Static" means g = 1.

## Estimators
| name | what it does |
|---|---|
| raw | unmitigated (readout-mitigated) value at λ=1 |
| PS | post-select on weight-3 sector, λ=1 |
| ZNE-lin / ZNE-exp | standard ZNE on the raw values vs nominal λ |
| **PS+ZNE-lin** | post-select, then linear ZNE vs nominal λ (**pre-registered comparator**) |
| PS+ZNE-rich | post-select, then Richardson (quadratic) vs nominal λ |
| GUESS-Pi | GUESS-style weights from the sector-projector decay |
| SA-v1-lin/exp | first attempt (failed) |
| SA-ZNE (primary, ε-lin) | PS observable linear in ε = 1−f, with f = (P_keep − p_rand)/(1 − p_rand) |
| **SA-ZNE-loglin** (refined) | PS observable linear in x = −ln f, extrapolated to x = 0 |

p_rand = 20/64 is the keep-probability of a fully depolarised 6-qubit state.

## History (honest)
1. **v1** (pilot): failed, MAE ≈ 0.052 vs PS+ZNE-rich ≈ 0.020. Exponential variant numerically unstable.
2. **Primary pre-planned SA-ZNE (ε-lin)** failed on the main run: MAE 0.031–0.092, +140–370 % worse than the best baseline.
   Diagnosis: least-squares linear in P_keep extrapolated to P_keep=1 is algebraically identical to GUESS-Pi with
   post-selection (min-norm weights with Σw=1, w·P=1), and the sector decay is far from linear in λ.
3. **Refinement SA-ZNE-loglin:** −ln f is (to first order) proportional to the effective noise g·λ,
   so it replaces nominal λ with a measured λ_eff. This was chosen **after** seeing the main data,
   so a held-out **confirmatory run** (seed 3031, new instances) was pre-registered in `PREREGISTRATION.md`.

## Results
### Main run (seed 2026; 20 instances × 3 depths × 2 observables = 120 problems per cell)
Mean absolute error (lower is better):

| cell | raw | PS | PS+ZNE-lin | PS+ZNE-rich | SA-ZNE ε-lin | **SA-ZNE-loglin** |
|---|---|---|---|---|---|---|
| pooled drift (480) | 0.0746 | 0.0259 | 0.0178 | – | 0.0597 | **0.0165** |
| pooled static (480) | 0.0747 | 0.0259 | 0.0163 | – | – | **0.0163** |

Pooled drift paired difference (loglin − PS+ZNE-lin) = **−0.00136, 95 % CI [−0.00197, −0.00075], −7.6 %**.
Static: −0.000001, CI [−0.00058, +0.00055] (parity).
Drift penalty (MAE_drift − MAE_static): loglin +0.00019 vs PS+ZNE-lin +0.00155 (difference CI [−0.00171, −0.00102]).

Per cell (drift): Torino/global −14.2 % (CI [−0.0028, −0.0012], Wilcoxon p = 1.9e-8); Sherbrooke/global −6.7 %;
Sherbrooke/local −6.0 % (CI crosses 0); Torino/local −5.0 % (CI crosses 0).
Losses: Torino static local +17 %; on Sherbrooke global, Richardson PS+ZNE-rich (0.0219 drift / 0.0197 static) beats loglin (0.0234 / 0.0232).
By depth (drift): 1 step 0.0125 vs 0.0116 (loss), 2 steps 0.0150 vs 0.0154, 3 steps 0.0219 vs 0.0265 (win). The benefit grows with depth.

### Confirmatory run (seed 3031; 10 new instances; 60 problems per cell) – pre-registered
| hypothesis | result | verdict |
|---|---|---|
| H1: pooled drift diff < 0, CI entirely below 0 | −0.00046, CI [−0.00108, +0.00018], −2.7 %, win-rate 0.48 | **NOT confirmed (falsified as stated)** |
| H2: static parity | +0.00030, CI [−0.00035, +0.00092], +1.9 % | consistent |
| H3: smaller drift penalty | 0.00082 vs 0.00158; diff −0.00076, CI [−0.00119, −0.00032] | **confirmed** |

Per cell (drift): Torino/global −14.8 % (CI [−0.0026, −0.0014]) replicated; the other three cells ≈ 0 (−0.1 %, +1.3 %, −0.6 %).

### Colab notebook demo (`notebook/error_mitigation_demo.ipynb`, FakeManilaV2, 4 qubits, 60 problems, ~40 s)
Steady / drifting MAE: raw 0.0724 / 0.0725; PS 0.0446 / 0.0447; standard ZNE 0.0282 / 0.0305; SA-ZNE 0.0335 / 0.0344.
SA-ZNE is **worse** here (+19 % steady, +13 % drifting) but loses less to drift (0.0009 vs 0.0023).

## Interpretation
- The robust, replicated finding is **H3: SA-ZNE-loglin is far less sensitive to drift between the λ circuits**
  (it loses 0.0002–0.0008 MAE to drift where the nominal-λ baseline loses ≈0.0016).
- The accuracy gain is **not general**: it replicated only on FakeTorino with global folding (≈ −14 %). On
  FakeSherbrooke the measured gauge under-tracks the true g·λ ratio (fit slope 0.36–0.42, vs 0.77–0.86 on Torino;
  `figures/fig3_gauge_tracking.png`), since much of the noise there (readout, amplitude damping) does not leave the symmetry sector.
- On small/static or readout-dominated devices (Manila demo, Torino local static) plain PS+ZNE-lin is better.

## Limitations
Simulation only (Aer noise models built from fake-backend snapshots); a toy drift model (scalar gain on gate errors);
small chains (n = 4, 6); one symmetry (U(1) magnetisation); only symmetry-*detectable* errors are gauged;
the refinement was post hoc (mitigated by the held-out run, which partly failed); the gauge needs enough kept shots (f not ≈ 0).

## Reproduce
```bash
python -m venv qenv && . qenv/bin/activate && pip install -r requirements.txt
./run_all.sh                       # ≈ 47 min main + ≈ 18 min confirm on 7 cores
# or step by step:
python run_experiments.py --instances 20 --steps 1,2,3 --seed 2026 --workers 7 --out results/probs.npz
python analyze.py --probs results/probs.npz --shot-seeds 20 --tag main
python pooled.py main && python make_figures.py main
```
The notebook: open `notebook/error_mitigation_demo.ipynb` in Colab and Run all (~1 min after installs).

## Files
- `sazne/` – circuits, observables, simulator (noise + drift + readout), estimators
- `run_experiments.py`, `analyze.py`, `pooled.py`, `make_figures.py`, `run_all.sh`, `PREREGISTRATION.md`
- `results/` – raw probabilities (`probs*.npz`), per-problem estimates (`estimates_*.csv`), summaries (`summary_*.json`, `pooled_*.json`, `analyze_*.txt`), logs
- `figures/` – fig1–fig5 for the main run; `figures/confirm/` – the same for the confirmatory run
- `notebook/` – Colab-ready demo, its builder, and an executed copy

## Prior art (closest)
- Koenig, Reinecke, Hahn, Wellens, *Inverted-circuit zero-noise extrapolation*, arXiv:2403.01608, PRA 110, 042625 (2024)
- Bonet-Monroig, Sagastizabal, Singh, O'Brien, *Low-cost error mitigation by symmetry verification*, arXiv:1807.10050, PRA 98, 062339 (2018)
- *GUESS: Guiding Extrapolations from Symmetry Decays*, arXiv:2603.13060
- Cai, *Multi-exponential error extrapolation and combining error mitigation techniques*, arXiv:2007.01265, npj QI 7, 80 (2021)
- Urbanek et al., *Mitigating depolarizing noise ... with noise-estimation circuits*, arXiv:2103.08591, PRL 127, 270502 (2021)
- Purity-assisted ZNE, arXiv:2310.10037; Physics-Inspired Extrapolation, arXiv:2505.07977
- Temme, Bravyi, Gambetta, arXiv:1612.02058; Giurgica-Tiron et al., arXiv:2005.10921; Kim et al., Nature 618 (2023)
