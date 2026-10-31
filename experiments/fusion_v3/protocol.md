# Fusion v3: Matched Causal One- versus Six-Hour Planning

Frozen before running v3 training/evaluation on 2026-10-26.
All source years and v1/v2 outcomes have already been inspected: exploratory,
not a new untouched holdout or a confirmatory preregistration.

## Question and Gate

Does genuine six-hour lookahead alleviate high-load recovery charging without
free energy, terminal failure, or hidden cycling costs? Separate planning-window
effects from predictor effects. Existing v2 one-hour control failed admission;
v1/v2 arithmetic/hash audits passed. Do not change the production application.

## Data, Issuance and Split Embargo

Use the same Germany hourly data and chronological boundaries as v1.
A row timestamp t is its first target hour. Its latest observation is t-1h;
the forecast is available at the start of t under the explicit idealized
zero-publication-delay assumption. issued_timestamp denotes t-1h, not a
claim that a completed hourly observation existed earlier.

Predict components total/solar/wind/net at target hours t through t+5h.
Use v1 causal historical features plus known future calendar sin/cos for each
extra horizon. No future measured weather, load, generation, or target residual.
Targets must be aligned by timestamps before removing missing rows.

Require ALL six finite labels within a split for its training/evaluation rows.
Training labels must end before 2019-01-12 20:00 UTC; validation labels before
2019-08-09 21:00 UTC; test labels remain within the original source extent;
2021/2022 labels remain in their respective target years. Drop boundary rows
and disclose counts. Calibration and dispatch use common complete UTC days.
These exclusions differ from v2, so no direct cross-round peak attribution.

## Frozen Predictors and Budget

- Direct net-load HGB for each of six horizons: learning_rate .08, 350 iterations,
  max_leaf_nodes 31, l2_regularization 1, random_state 20261026.
- Joint six-horizon residual MLP, two 64-unit SiLU layers, 24 outputs.
  Latest observed four components are the residual anchor at every horizon.
- Paired data/physics arms for seeds 42,43,44, identical initialization/batches.
  25 epochs, batch512, AdamW lr .001/weight_decay .0001, gradient clip5.
  data: net MSE + .25 component MSE; physics adds .2 accounting MSE and
  .1 negative-component penalty, averaged across all six horizons.
- Train-only feature standardization; choose checkpoint by validation mean
  net-load MAE across all six horizons, not test outcomes.
- Persistence repeats the latest OBSERVATION, explicitly a naive baseline,
  not a fabricated multi-hour neural forecast.
- Raw outputs feed dispatch. Projection is recorded only as forecast diagnostic,
  not another dispatch sweep. No seed or horizon selected by test performance.

## Matched Controller and Calibration

Eight predictors (HGB, persistence, six MLPs), three evaluation periods, and
two planning windows: 48 nominal traces. Within each predictor, BOTH windows
use exactly the same h1 forecast, h1 radius, gates, samples and uncertainty
scale (1.1 times the maximum radius across ALL six horizons/hours, min1).
Six-hour dynamic reserve additionally sees future calibrated bands, an explicit
part of the existing controller, not an isolated LP-lookahead causal claim.

Reuse simulate_balanced_storage unchanged with point objective and daily
terminal equality. Common actual validation q95/q25/q99 gates. Each method's
absolute-error q90 is calibrated by horizon and target UTC hour, using only
complete validation rows whose last target is before test start.
Capacity30000 MWh, initial/operational ceiling/terminal15000 MWh,
minimum4500 MWh, powers1250 MW each, per-direction efficiency .92.
No free energy reset across consecutive days. Preserve all recovery charges.

## Stress and Metrics

Fixed probe data_seed42, six-hour panel, 2021/2022. Same two parameter changes
as v2: both powers875 MW, or efficiency .85. For each compare continuous
nominal-action replay with parameter-aware six-hour replanning, parameters
perfectly known at episode start. Eight paired traces, no health-sensing claim.

Report every seed: h1..h6 MAE and accounting diagnostics; controlled maximum,
P95, mean daily peak, recovery/high-load charging, throughput, conversion loss,
terminal deficits, SOC/power/continuity/energy audits and run failures.
Primary contrasts: 6h minus1h within predictor; physics minusdata by paired seed
at each window; raw data MLP minusHGB at each window. Daily-peak intervals
use fixed-seed averages and seven-day circular paired blocks, 2000 replicates,
RNG20261026, conditional on observed days and fixed seeds. Do not average
away max-peak failures or call interval overlap a proof of equivalence.

## Execution and Verification

Use one free RTX3090 sequentially for six small MLPs; inspect occupancy first.
HGB single-thread CPU; dispatch at most three workers, one numerical thread each.
No environment installs, external services, unrelated process termination or
changes to old source/result hashes. Refuse to overwrite outputs/fusion_v3.
Record data/source/protocol/checkpoint/input/artifact hashes and package versions.

Preflight tests: causal mutation, correct horizon labels, split embargo, validation
lower/upper bounds, future-label-invariant inference, 1h/6h h1 identity,
no actual-load influence on actions, physical audits, and timestamp-unit-safe
720-hour rolling intervals. Independently recompute delivered CSV results locally
and reload a saved neural checkpoint on remote CPU. Leave failures visible.
