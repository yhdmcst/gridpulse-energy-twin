# Fusion v6: Delayed Information and Model-Plant Mismatch

Frozen before v6 simulation on2026-10-26. All years and v1-v5 outcomes have been
inspected previously. Exploratory stress audit; no new untouched holdout.
Keep frozen v5 selected policy tail_guard; no new policy selection or training.

## Questions and Scope

1. Does tail_guard retain its matched legacy advantage with1h/3h delayed grid
   observation/forecast availability?
2. How much is information age versus the shorter remaining forecast horizon?
3. If the controller assumes nominal powers/efficiency but the plant is worse,
   do clipped actions and continuous energy still meet terminal requirements?
4. Is the evidence sufficient for a real-time/default demonstration interface?

No UI edits before deciding. This is not a fault detector, online system
identifier, delayed-SOC-feedback experiment or field deployment test.

## Frozen Predictors, Calibrations and Common Samples

Data MLP seeds42/43/44 only,2021 and2022. Keep the same source truth, v3 raw six-
horizon forecasts and v5/v4 validation gates/tail threshold/calibrated radii.
No new residual calibration on evaluated outcomes.

Each source forecast row has first target s, latest observation s-1h, and six
targets s..s+5h. At target t with extra publication delay d, only the forecast
row with s=t-d is used: relative horizon h takes original horizon d+h.
Source issued_timestamp=t-d-1h; available_timestamp=s+d=t. Do not shift actual
loads, use the currently issued forecast, repeat unavailable h1, or fabricate
new neural forecasts. Bounds use original lead d+h and target UTC hour; retain
the same uncertainty_scale from all six original leads.

Five information modes:
- fresh6: delay0,6h;
- fresh5: delay0,5h;
- fresh3: delay0,3h;
- delayed1: delay1,5h;
- delayed3: delay3,3h.

Intersect timestamps across every mode and all three seeds within each year,
then keep common complete UTC days. Source boundary exclusions are explicit.
All within-round contrasts use the same hours. Typical expected sample is
Jan2-Dec30 each year because no older origin exists before source Jan1.

## Delay Comparisons and Replication

Two policies legacy/tail_guard x five modes x three seeds x two years =
60 policy traces, organized as30 paired jobs.
Same nominal physical model as v5:30000MWh capacity,15000 initial/ceiling/terminal,
4500 hard minimum,1250MW both powers,efficiency.92 per direction.
Battery SOC is current and exact; only grid observations/forecasts are delayed.
Reference is recomputed from the same available panel, not a fresh-information
reference. Tail_guard reads only the current reference state/action as in v5.

For fresh6 require exact numerical replication (<=1e-5MW/MWh action/state
difference) of v5 trajectories restricted to the same complete days.
This adapter check is a gate before interpreting stress results.

## Parameter-Error Probes

Fixed seed42, both years, fresh6/delayed1/delayed3.
The planner assumes nominal.92 efficiency/1250MW powers in each error case;
the actual plant instead has one of:
- power_90pct:1125MW both directions,efficiency.92;
- power_70pct:875MW both directions,efficiency.92;
- efficiency_090:efficiency.90,powers1250MW;
- efficiency_085:efficiency.85,powers1250MW.

Both policies' frozen nominal requested actions are executed in actual physics,
clipped by plant power and SOC limits, carrying actual SOC continuously across
days without reset.48 mismatch traces. This is FIXED-REQUEST robustness,
not online SOC-feedback correction. Power clipping that protects hard bounds
does not make terminal failure disappear. Requests and unexecuted energy stay visible.
Only optimistic model errors are tested; do not generalize to arbitrary error signs.

For each same condition, recompute legacy reference and tail_guard with perfectly
known actual parameters and the same delayed forecasts:48 known-parameter policy
traces,24 additional paired jobs. These are labelled known-parameter benchmarks,
not automatic adaptation. Total156 traces:60 delay+48 mismatch+48 known.

## Metrics and Paired Evidence

Every trace: actual/no-storage max/P95, controlled max/P95, mean daily peak,
throughput, conversion loss, recovery charging where defined, actual-high-load
charging, terminal maximum/error hours/days, power/SOC/continuity/conservation.
Delay traces additionally h1 MAE, actual band coverage, issuance age, horizon,
reference debt and guard diagnostics.
Mismatch traces additionally clipped/unexecuted charge/discharge and SOC-limited hours.

Paired tail_guard-minus-legacy contrasts in every condition; freshness contrasts
delayed1-minus-fresh5 and delayed3-minus-fresh3 separate information age.
Fresh5/fresh3-minus-fresh6 describe horizon effect at fixed forecast freshness.
Daily-peak difference intervals:7-day circular paired blocks,2000 replicates,
RNG20261026, conditional on observed days/fixed models; no field-risk guarantee.

## Demo Admission Decision (Not a New Optimizer)

Save this rule before simulation. A default/real-time candidate demonstration
is admitted ONLY if all of the following hold:
- no failed jobs and all independent timing/physical/replication audits pass;
- for each year and main mode fresh6/delayed1/delayed3, every seed meets the
  unchanged matched-legacy guards: maximum<=legacy+25MW, P95<=legacy+1MW,
  throughput/loss<=1.10*legacy+1e-6, zero terminal error within1e-5MWh,
  and fixed-seed mean P95 improves by at least1MW;
- all24 tail_guard mismatch replays pass the daily terminal requirement within
  1e-5MWh as well as hard physical bounds.

Any failure defers default/real-time UI integration. Do not relax after results.
Known-parameter success cannot override mismatch failure. A fully labelled offline
stress-results viewer may be scientifically appropriate but is NOT implemented
or promoted to live control in this request. Record individual failed conditions.
Perfect terminal tolerance is the same stated daily equality, not a field regulation.

## Implementation and Verification

Reuse existing SciPy HiGHS solvers and frozen recovery solve_window, with a new
variable-horizon adapter only. Refuse overwrite outputs/fusion_v6. CPU at most3
workers/one numerical thread each; no GPU, installs or unrelated task interference.
Record all code/protocol/data/forecast/input hashes and failures.

Preflight: exact old lead alignment and interval selection, no stale-row boundary
interpolation, current/future truth invariance, newly issued forecast mutation
cannot alter a delayed action, reference causality, generic horizon physics,
continuous replay without resets, mismatch terminal failure remains failure,
matched fresh6 replication, and deterministic demo-rule failure/pass examples.
Independently reconstruct delayed panels and all metrics locally, verify mismatch
actions from physical equations and confirm decisions; rerun v5 integrity.
