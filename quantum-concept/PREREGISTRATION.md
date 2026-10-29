# Confirmatory test (written 2026-10-29 14:57 GMT, while the held-out run was in progress, before its results existed)

The exploratory run (seed 2026, 20 instances) showed that the pre-registered primary SA-ZNE
(linear in eps = 1 - f) FAILED: it is algebraically identical to GUESS-Pi+PS and 2-4x worse
than the best baseline. The secondary variant SA-ZNE-loglin (linear in x = -ln f) looked
better under drift. Promoting it is a post-hoc choice, so it is re-tested on fresh data:

- Data: run_experiments.py --instances 10 --seed 3031 (new random XXZ instances); same backends,
  depths, drift gains and shot settings. Analysis: analyze.py --tag confirm; pooled.py confirm.
- Fixed comparator, chosen now and not post hoc: PS+ZNE-lin (the best nominal baseline in 6 of 8 cells).
- H1 (drift): pooled over backend x folding scheme, the mean paired difference
  MAE(SA-ZNE-loglin) - MAE(PS+ZNE-lin) < 0, with the 95% bootstrap CI entirely below 0.
- H2 (static): parity is expected, not an advantage.
- H3 (robustness): the drift penalty (MAE_drift - MAE_static) of SA-ZNE-loglin is smaller than
  that of PS+ZNE-lin, with the paired CI below 0.
The claim is falsified if the H1 CI includes 0 or lies above it.
(Note: the box was restored from a snapshot at 14:55 GMT, and this file was rewritten with the same content.)
