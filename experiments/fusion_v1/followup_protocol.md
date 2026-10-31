# Followup: Validation-Calibrated Monitor and Causal Intervals

Written after inspecting v1. Original heuristic monitor alarms in approximately
94-97% of unmodified historical windows. Keep original results; do not retune
that reported experiment or call unmodified periods labelled normal.

Use the fixed data-only seed 42 model (not selected by test performance).
Training residuals determine mean/std/mean absolute ramp reference and scale.
Nonoverlapping complete 168-hour validation windows determine a fixed threshold
at finite-sample upper 95% rank. At least 20 windows required. No evaluation
outcome enters fitting. Window score is maximum absolute change of the three
statistics divided by the training residual standard deviation.

Evaluate test/2021/2022 unmodified windows and the SAME windows plus deterministic
3000 MW residual bias. These labels are synthetic only. Report unlabelled
historical alarm rates, not field FPR. No first-detection-time or device-fault
claim.

Separately compare fixed validation q90 intervals and rolling q90 absolute
residual intervals from the preceding 720 hours, requiring 168 prior outcomes.
At target hour i, its own residual and all future residuals are excluded.
Fallback is fixed validation radius. Additional publication latency is assumed
zero as in the original project. Evaluate coverage, width AND interval score;
do not select the controller from these diagnostics.

No new model training, checkpoint selection, physical tuning or controller
adaptation. This is an exploratory calibration correction, not independent
confirmation.
