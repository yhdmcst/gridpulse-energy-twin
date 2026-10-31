# Fusion v5: Causal Reference-Relative Recovery Governor

Frozen before v5 simulations on 2026-10-26. All years and v1-v4 outcomes
are already inspected. Exploratory engineering, not a confirmatory new holdout.

## Review and Question

V4 arithmetic and conservation passed but every new controller failed admission.
Local forecast peak caps did not prevent realized global peak increases.
Different SOC trajectories motivate, but do not prove, an energy-debt explanation.
Test whether a reference-relative recovery governor enables useful tail control.
Do not relax the v4 admission guard or select on evaluation outcomes.

## Fixed Forecasts, Reference and Physics

Reuse v3 raw six-hour HGB and data MLP seeds42/43/44, same complete UTC days,
v4 validation q90 tail threshold, q95/q25/q99 gates, bands and uncertainty scale.
Nominal model remains capacity30000MWh, initial/ceiling/terminal15000MWh,
minimum4500MWh, powers1250MW each, per-direction efficiency.92.

Reference is the v4 legacy six-hour policy. Its current action uses the same
current six-hour forecast and its state only integrates its past actions.
At hour t, use ONLY the reference's state-before-t, action-t and state-after-t.
No future reference actions/states, actual outcomes or later issued forecasts
may enter the candidate's current optimization. Reference input strips actual
loads and unrelated fields. Reuse16 frozen legacy CSVs byte-for-byte.

The existing instantaneous observation-availability assumption still applies.
Missing or invalid reference/forecast inputs must fail, not trigger free SOC resets.

## Five Declared Controllers

1. tail_unanchored: single-stage mean forecast excess above the fixed validation
   tail threshold plus .001*throughput; original dynamic reserve and reachability.
2. tail_debt: same plus first-step SOC >= current reference state-after-t.
3. tail_guard: same debt floor plus a first-action grid-power upper bound.
4. peak_debt: window max point forecast plus .001*throughput, plus debt floor.
5. peak_guard: same peak objective plus debt floor and grid-power upper bound.

For guard policies require:
    candidate_charge_t - candidate_discharge_t
      <= max(reference_charge_t - reference_discharge_t,
             validation_tail_threshold - current_upper_h1).

At forecast-upper-high hours this prevents an increased net storage load
relative to reference. Otherwise it permits charging only within band headroom.
It is NOT a guarantee on actual load because a calibrated band can miss reality.
With zero SOC debt the feasible reference action (charge clipped to candidate
headroom if needed) motivates feasibility under the same physical parameters.
Still audit plans and executed actions numerically; no general field proof.

Debt is constrained ONLY at the next executed state. Future window SOC bounds
retain existing hard bounds, dynamic advisory floor and daily terminal
reachability; never use the future precomputed reference trajectory.
Daily terminal remains equality. No above-reference initial energy is injected.

Unlike v4 tail controllers, the tail-only objective does NOT first minimize
local peak. The unanchored control isolates this objective change; tail_debt
then isolates the SOC floor; tail_guard isolates the additional headroom cap.
Peak controls provide an objective ablation, not a claim of perfect legacy identity.
Allow a negative max variable, remove simultaneous actions preserving energy
and decreasing net storage load, and audit the resulting plan.

## Selection, Locked Budget and Evaluation

Six policies including legacy x four predictors =24 validation traces,
of which4 legacy references are reused. Five x4 =20 new validation simulations.
Use only the THREE data seeds for one global selection; HGB is a transfer check.
Same v4 guards per seed:
- physical/terminal audits pass;
- maximum <= legacy maximum+25MW;
- throughput/loss each <= legacy*1.10+1e-6MWh;
- P95 <= legacy P95+1MW.
Mean P95 improvement must be at least1MW; minimize mean P95, break ties by
mean loss then policy name; fallback legacy if none. All five candidates
participate under the same rule; do not force the governor to win.

Save selection receipt/input/protocol hashes and freeze UTC time BEFORE parsing
known evaluation-period outcomes or reference rows. Calibration/selection reuse
and prior year inspection remain explicit limitations.

Evaluate ALL six policies x4 predictors x3 periods =72 evaluation traces,
of which12 legacy are byte-reused. No evaluation-based reselection or grid expansion.
Report each seed, no-storage/max/P95, daily peak, fixed-threshold mean excess,
throughput/loss/recovery/high-load charging, debt and cap diagnostics.
Paired differences against legacy for all60 new runs; fixed-seed means for
the selected control. For daily mean peak/excess use seven-day circular paired
blocks,2000 replicates,RNG20261026. Intervals are conditional, not independent grids.
P95 changes remain descriptive and not a field reliability statement.

## Stress

Fixed data_seed42 selected-policy probe,2021/2022,875MW powers or efficiency.85.
Compare continuous nominal-action replay and known-parameter replanning:
8 traces. Replanning recomputes the causal legacy reference with actual changed
parameters, not an infeasible nominal reference. Save and audit four changed
reference traces in addition to these8. Parameters known from episode start,
no health inference or sudden-unknown-fault claim.

## Provenance and Preflight

New outputs/fusion_v5 only. CPU only, at most3 workers, OMP/BLAS threads1.
No installs, retraining, GPU allocations or prior source/result modifications.
Record source/protocol/input/artifact hashes and every failure.
Preflight: truth invariance, valid future-issued-forecast mutation changes no
earlier actions, reference alignment/physics, no future reference access,
zero debt and cap enforcement, boundary equality, negative-load handling,
selection guards/fallback, parameter-aware reference changes and canonical actions.
Independently recompute delivered physics, selection, reference states and
headroom/debt bounds locally; rerun earlier immutable audits.
