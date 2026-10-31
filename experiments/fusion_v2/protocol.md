# Fusion v2: Forecast-to-Dispatch and Parameter-Aware Twin Replanning

Locked before executing the new dispatch comparisons. The v1 predictions and
the historical evaluation data are already known. Exploratory extension only.

## Questions

1. Does lower one-hour forecast MAE translate into lower controlled peaks/P95/
   mean daily peaks under matched physical constraints?
2. Does accounting regularization/projection help downstream operation?
3. Under known power/efficiency changes, can twin-aware replanning restore the
   end-of-day energy budget that frozen-action replay violates, and at what cost?

## Inputs and Sample

Reuse all frozen v1 outputs: persistence, seasonal24, HGB, and data/physics MLP
raw/projected outputs for seeds 42/43/44. No new model training or seed selection.
Intersect ordered timestamps and actual net loads across all methods; retain
only complete UTC days. Use the same complete validation days for every method.
Evaluate original test and 2021/2022 separately, preserving all failures.

## Controller

Use existing `simulate_balanced_storage` / SciPy HiGHS LP with ONE forecast hour.
Do not repeat an h1 forecast as a six-hour prediction or use future target truth.
Same point-objective dynamic-reserve controller, power and energy equations.

Common gates use validation ACTUAL net-load q95 (peak), q25 (charge), q99
(extreme release), computed once across the common complete validation days.
For each method/seed, validation absolute-error q90 by UTC target hour determines
bands; uncertainty scale is 1.1 times the maximum calibrated radius (at least 1 MW).
This compares predictors plus their calibration pipeline, not MAE in isolation.

Nominal model: capacity 30000 MWh, operational ceiling/initial/terminal 15000 MWh,
minimum 4500 MWh, both powers 1250 MW, efficiency .92. Include recovery charging
in load and losses. Every independent UTC day starts/ends at 50%; continuity
across consecutive days must also be verified, not reset away a deficit.

## Metrics and Comparisons

All 15 methods x 3 periods: peak, P95, mean daily peak, throughput, conversion
loss, h1 MAE, power/SOC/continuity/energy/load/terminal errors. Report every seed.
No-storage baseline uses the same actual hours; preclude free initial energy.

Paired contrasts: data raw vs HGB, data projected vs data raw, physics raw vs
data raw, physics projected vs data projected. Use fixed-seed averages, not
best seed. Seven-day paired circular blocks, 2000 replicates for mean daily
peak difference, conditional on observed days and these fixed trained seeds.
Do not label seeds or hours as independent new grids.

## Paired Stress Experiment

Fixed method `data_projected_seed42`, chosen for this engineering probe, not
by evaluation outcomes. For 2021 and 2022 compare:

- common power derating to .70 times nominal power;
- efficiency decrease from .92 to .85.

For each change, both arms have the same actual physics, starting SOC, forecasts,
bands, timestamps and gates. Replay arm executes nominal requested actions
under changed physical bounds, carrying SOC continuously without daily reset.
Replan arm recomputes actions using the changed parameters known at episode start
and daily-reachability constraints. Perfect parameter knowledge is an explicit
assumption; no health estimation or sudden unobserved-fault guarantee.

Compare terminal deficit, clipped/unexecuted action, daily peak, P95, throughput
and loss. Satisfying bounds by construction does not establish field safety.
Replanning may worsen peaks and increase charging costs; retain these outcomes.

## Provenance and Safety

New output directory only: `outputs/fusion_v2`. No original or v1 overwrites.
Record input/code/protocol hashes, package versions, complete-day exclusions,
calibration values, per-hour traces, seed tables and failure statuses.
At most 3 CPU worker processes, one numerical thread per worker; no GPU
allocation is needed for already generated forecasts and small LPs.
Before execution require tests proving no current/future truth enters actions,
validation-only calibration, strict missing-data behavior and independent
conservation/terminal checks. Preserve failed runs and do not drop failures.
