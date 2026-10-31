# Fusion v7: Hourly Plant Feedback and Auditable Degradation

Frozen before v7 simulation on2026-10-26. Source years and v1-v6 outcomes already
inspected. Offline closed-loop simulation only; no UI or deployment change.
No retraining, parameter estimation, hyperparameter selection or new holdout claim.

## Closed-Loop Separation

At hour t the controller receives the actual physical simulator SOC before
choosing its request. After request execution the actuator applies true power,
efficiency and hard SOC limits, and returns the next SOC. True plant parameters
are NEVER arguments to the decision function. Net-load truth is scoring only.
Initial SOC15000MWh is supplied once per episode, never reset at midnight.

Planner remains1250MW powers/.92 per direction. Capacity30000MWh,
hard minimum4500MWh and operating maximum/target15000MWh remain known.
Battery SOC telemetry is exact and current; no physical battery, telemetry noise
or SOC communication delay is simulated.

A virtual legacy reference is propagated causally with nominal model parameters.
Its current modeled action and next modeled SOC may enter the existing v5
single-window recovery LP. Future reference rows are not read. The candidate
window uses measured plant SOC, not a nominal accumulated state.
If reference/deadline constraints cannot be met, record the optimization failure
and execute the declared fallback; do not claim the old reference-debt guarantee
held after actual model mismatch.

## Two Predeclared Variants

- feedback_only: measured plant SOC replan every hour, plus recovery fallback
  only on stale/invalid data, reference/primary optimization failure or outage.
- supervised: same controller and fallback, plus a one-step deadline supervisor
  on every request, before physical actuation.

The supervisor knows a DECLARED certified lower envelope P_charge>=875MW,
per-direction efficiency>=.85, not the actual per-scenario values.
For current hour h, remaining after action a=23-h, define
    floor_next=max(4500,15000-a*875*.85).

If SOC<floor_next, disallow discharge and ensure enough requested charge at
eta=.85, capped by875MW when recovery is unreachable. Otherwise limit requested
discharge to (SOC-floor_next)*.85. Existing requested charging may remain larger;
its guaranteed delivered contribution uses min(request,875)*.85.
Compute/log worst-case next SOC, intervention and unreachable-recovery flags.
Hardware may clip at the upper SOC boundary; that is not free injected energy.
When bounds are valid and the invariant holds initially, this motivates a
deadline invariant, but still numerically audit every step.

Recovery supervision has priority over the forecast headroom cap. Report any
headroom relaxation, extra charging, new peaks, throughput and loss. Do not claim
the original v5 strategy is unchanged when the supervisor/fallback overrides it.

## Degradation Rules (Frozen)

Maximum allowed grid-observation age at the current hour:2h, including the
previous-hour measurement lag. Thus fresh6 and delayed1 are accepted;
an old delayed3 packet (age4h) is expired.
Use frozen v6 panels and their real old-origin forecasts/bands; do not retimestamp
fresh forecasts as stale. Availability in the future or invalid bands are rejected.

Expired/invalid packet or deterministic solver outage:
- do not call the optimization backend;
- do not reuse an old charge/discharge instruction;
- request conservative recovery only if SOC is below floor_next, otherwise idle;
- log reason, current SOC, floor, request, actual action and recovery deficit.

For natural window RuntimeError (infeasible/refinement failure), use the same
fallback and retain the error/reason. Unexpected programming/validation errors
remain failed jobs, not successful degradation.
The virtual reference follows its own nominal physical evolution and uses the
same data/service gate; it is not reset or replaced by future actual states.

## Data, Conditions and Counts

Same common v6 hours:Jan2-Dec30 in2021/2022,363days/8712h each.
Fixed data_seed42 only; robustness probe, not a three-seed generalization study.
Fresh6 and delayed1 use the same v6 forecasts/calibration/tail gates unchanged.
Actual models are constant per episode:
- nominal1250MW/.92;
- power90:1125MW/.92;
- power70:875MW/.92;
- efficiency090:1250MW/.90;
- efficiency085:1250MW/.85;
- combined90:1125MW/.90.

Two years x two information modes x six plants x two variants =48 healthy
closed-loop traces. Nominal feedback_only must reproduce v6 tail_guard on the
same hours,<=1e-5MW/MWh action/state/load difference (4 checks).

Fault probes:combined90 plant,fresh6, both variants and years:
- stale_burst:every seventh episode day (day_index%7==0),UTC14..17 substitute
  the valid old-origin delayed3 packet; the age gate must reject it;
- solver_outage:the same days atUTC10, bypass BOTH virtual-reference and
  primary optimization, using recovery fallback;
- combined_fault:both schedules, disjoint hours.
12 fault traces. Schedules use calendar only, never load extrema.

Out-of-envelope diagnostic:750MW/.80, fresh6, supervised, both years:2 traces.
Outside the declared recovery knowledge, retain failure instead of broadening
the guarantee. No selection uses this outcome.

Open-loop comparator:24 traces (two years,two modes,six models) using exactly
the v6 nominal tail_guard requests in real plant physics, continuous SOC.
Reuse20 v6 nominal/error comparator CSVs byte-for-byte and compute4 combined90
replays. Label all open-loop comparisons distinctly.

Also save10 modeled references (four healthy context streams and six fault
streams). Total96 physical/state trajectories:62 closed+24 open+10 reference.

## Metrics and Decision Scope

Every trace:actual/no-storage peak/P95,daily peak,throughput,conversion loss,
continuous SOC,terminal error/failure days,actual clipping and unexecuted energy.
Closed-loop traces:measured SOC at decision,reference prediction/debt,solver
attempt/failure,expiry/outage/fallback reasons,supervisor interventions,
worst-case floor/invariant errors,headroom override and recovery unreachable.
Never count fallback or hardware clipping as evidence of forecast optimality.

Compare feedback_only and supervised to the SAME open-loop plant/mode, and
fault probes to combined90 healthy with the SAME variant. Use paired daily
peak circular7-day blocks,2000 replicates,RNG20261026; conditional only.
No new policy winner, guard-budget relaxation or evaluation-year selection.

Record separate capability checks:
1.nominal replication and state/timing/physics audit;
2.fault gates match the declared calendar and never call optimization while gated;
3.supervised in-envelope runs satisfy actual and worst-case floor/terminal checks.
Even if all pass, realtime_default_admitted remains false: this is a simulated,
perfect-SOC, assumed-envelope lab. Offline labelled replay only. No UI edits.

## Execution and Verification

New outputs/fusion_v7 only; refuse overwrite. Existing remote environment,
CPU at most3 workers,OMP/BLAS threads1. No GPU/installs/driver changes.
Record all source/protocol/input/artifact hashes and preserve failed jobs.

Preflight:actual-load invariance,unknown-parameter first-request invariance,
measured-state sensitivity,future-packet prefix invariance,no midnight reset,
stale/outage zero optimizer calls,no cached discharge,reference-only model
evolution,supervisor algebra and out-of-envelope failures,wrong bounds rejected.
Independently recompute actuator equations,feedback equality,TTL/fault times,
fallback/supervisor formulas,nominal replication,metrics and capability decisions
locally. Rerun v6 immutable audit before reporting.
