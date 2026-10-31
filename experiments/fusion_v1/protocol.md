# Fusion v1: Accounting-Constrained Net-Load Forecasts

Status: locked design before new model training. Existing project test results
and 2021/2022 data have already been inspected. This is an exploratory extension,
not a new preregistration or untouched confirmatory holdout.

## Question

Does accounting-consistency regularization improve one-hour net-load forecasts
relative to an otherwise identical residual neural network?

## Scope Correction

Chinese Open-Meteo/PVGIS files contain weather/resource estimates, not measured
plant output. They cannot establish a PV-power prediction gain. They are not
joined to German data and are not used as power ground truth in this study.
No IGBT, 4DGS, online retraining, live dispatch or financial-benefit claim.

## Design

- Targets: measured total load, solar, wind and their defined net load, in MW.
- Information: lagged observations only (1/2/3/24/25/168 hours), lagged 24-hour
  summaries and known target calendar features. No future weather/observations.
- Train: before 2019-01-12 20:00 UTC.
- Validation: through 2019-08-09 20:00 UTC.
- Original test: 2019-08-09 21:00 UTC to end of supplied OPSD snapshot.
- Later evaluations: supplied 2021 and 2022 Energy-Charts snapshots. Each uses
  its own preceding observed hours; initial 168 hours and incomplete-feature
  rows are excluded, identically for all methods. Source/provider shift noted.
- Missing observations remain missing; no interpolation across partitions.
- Feature scaling fitted on training rows only.
- Residual MLP: 64/64 hidden units, SiLU, four outputs added to latest observation.
- Both arms supervise net load plus 0.25 times component MSE, MW divided by 10000.
- Data arm: no consistency or nonnegative penalty.
- Physics arm: add 0.2 times squared accounting residual and 0.1 times squared
  negative-component penalty, same network/initialization/batches/budget.
- Seeds: 42, 43, 44, AdamW lr=0.001, weight_decay=0.0001, batch=512, 25 epochs.
- Select each checkpoint by validation net MAE only; no test tuning.
- Physics raw and physics projected outputs both reported. Projection is an
  equal-weight Euclidean projection onto the accounting hyperplane followed by
  component nonnegative clipping and net recomputation. This guarantees only
  that defined relation, not optimal prediction or real-grid feasibility.
- Projection-only ablation also applied to the data arm without retraining.
- Baselines: persistence, previous-day and a new 350-iteration histogram
  gradient-boosting model on the identical engineered inputs/evaluation rows.
- No claim this accounting identity is a novel physical law or a PINN.

## Evidence

Save environment versions, GPU visibility/allocation, code/data SHA-256, run
config, initial model hash, per-epoch training/validation log, selected checkpoint,
per-timestamp predictions, MAE/RMSE/tail/interval and accounting-violation metrics.
Report per-seed values. Paired seven-day block bootstrap describes conditional
observed-day uncertainty, not independence of training seeds or field reliability.

## Replay and Drift

Replay original daily traces with declared 15000 MWh operating ceiling,
4500 MWh floor, 1250 MW powers, efficiency 0.92. Join actual solar on exact UTC
timestamps before applying a 30% solar drop (no fixed +1500 MW proxy). Keep
original requested actions; no claim of controller adaptation. Carry state
continuously, no free daily SOC resets. Report SOC clipping, lost requested
actions, terminal deviation and conservation checks as well as peak/cost.

Drift prototype uses a training-only reference and complete 168-hour observed
residual windows. Evaluate unmodified historical windows (unlabelled reference
alarm rate, NOT measured field false-positive rate), deterministic +3000 MW
residual injection (synthetic label only), and a causal rolling interval
alternative using only past observed residuals. Thresholds not test-tuned.

## Admission

Do not replace the frozen predictor or controller on test results. Keep a
negative result if the constraint only helps consistency, worsens prediction,
fails a seed, or loses to gradient boosting. Test timestamps and hashes must
match between arms. Exclude failed runs from no denominator silently.
