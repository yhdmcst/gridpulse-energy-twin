# Fusion v8: Planning-Time Recovery Envelope

Frozen before v8 numerical validation on2026-10-28. All source years/v1-v7
outcomes already inspected. Exploratory offline study only.

## Question

Does putting the supplied conservative recovery knowledge into the rolling
planner reduce the peak/P95 cost of v7 nominal planning plus posthoc supervision?
No training, price optimization, automatic system identification or UI changes.

## Fixed Policies and Models

Frozen predictor:data_seed42 six-horizon v3 MLP, same v6 forecast panels and
v4/v5 calibration, current exact simulator SOC and original expiry/outage gates.

Baseline:byte-identical v7 supervised traces, nominal1250MW/.92 planner/reference
followed by declared875MW/.85 recovery supervisor.

Two new policies:
- envelope_reference:legacy peak planner applied at measured current SOC with
  conservative875MW BOTH powers and efficiency.85, plus the same supervisor.
  This is the reference-style objective ablation, not a tail neural model.
- envelope_tail:existing v5 tail_guard LP, measured actual SOC and a virtual
  legacy reference BOTH using conservative875MW/.85, plus the same supervisor.

Use existing v7 reference_decision, candidate_decision, actuate, supervisor,
packet_gate, schedules and audit conventions. Do not alter frozen modules.
Both new policies use the SAME conservative virtual reference stream for
reporting; only envelope_tail imposes reference-debt/headroom in its primary LP.
All primary failures and supervisor overrides are visible.
Nominal baseline reference is generated separately on validation only.

This changes internal dynamics, request-power limits and virtual reference
together. It does not isolate one constraint, and can reduce discharge capability.
Conservative state predictions are not actual plant state; actual efficiency
greater than.85 may reach the operating ceiling and clip charging.
True per-scenario parameters appear only in physical actuation and scoring.
Supplied lower bounds are not certified from real equipment here.

## Validation and Frozen Retention Rule

Use shared complete validation days from v3 original validation predictions,
fresh6 only. All six targets before2019-08-09T21:00Z and after the validation
lower boundary2019-01-12T20:00Z. No evaluation residual recalibration.
Run nominal and combined90 actual plants for each of the two new policies
and v7 baseline =6 validation traces, plus2 modeled reference streams.

Retain at most ONE global candidate using ONLY these validation metrics.
For BOTH actual scenarios individually require:
- hard physics/terminal requirement pass within1e-5MWh;
- actual/worst-case recovery-floor errors<=1e-5MWh, gated optimizer calls0;
- maximum<=v7 supervised+25MW;
- P95<=v7 supervised+1MW;
- throughput and conversion loss<=1.10*v7 supervised+1e-6MWh.
Also require mean P95 delta over the two paired scenarios<=-1MW.
Choose lowest mean P95; tie mean conversion loss then policy ID.
If none eligible keep baseline_v7. No post-evaluation reselection, expanded grid
or relaxed guard. Save selection/input/protocol hashes BEFORE parsing v7
evaluation metrics or loading evaluation forecast panels.
Validation reuse and single fixed seed limit inference.

## Locked Evaluation

Use same2021/2022 v6 days:Jan2-Dec30,363days/8712hours each.
Do not tune with their already-known outcomes.
For BOTH new policies run:
-2years x2modes(fresh6,delayed1) x6 actual plants from v7 =48 healthy traces;
-2years x3events(stale_burst,solver_outage,combined_fault) xcombined90/fresh6 =12fault traces;
-2years xoutside(750MW/.80)/fresh6 =4 outside traces across2policies.
64 new evaluation traces. Keep ALL policies even if not retained.
Use v7 supervised same-condition baseline for32 conditions; reuse baseline
reference CSVs/metrics byte-for-byte at the trajectory level.
10 new conservative reference streams, validation6+references2.
Total114 trajectories:6validation+2validationrefs+64new evaluation+
32reused evaluation baseline+10evaluation refs.

Actual plants:nominal,power90,power70,efficiency090,efficiency085,combined90,outside,
exactly v7 definitions. Fault schedules and observed-age threshold unchanged.
No midnight reset, extra initial energy or future reference inputs.
Outside-envelope failures do not invalidate the bounded condition audit but must
remain reported; success outside bounds does not broaden the guarantee.

## Metrics and Evidence

Every trace:actual physics/continuity/load accounting,terminal errors/failure days,
max/P95,daily peak,throughput/loss,primary solver calls/failures,stale/outage
fallback,supervisor interventions,actual clipping,headroom violations where
applicable,and conservative predicted-versus-actual SOC.
New-policy minus matched v7 supervised on every condition.
For paired daily peaks use7-day circular blocks,2000replicates,RNG20261026;
conditional inspected days/fixed seed, not independent grids. No P95 p-values.
Keep selected designation fixed regardless of later-year tradeoffs.

## Preflight and Execution

Isolated outputs/fusion_v8; refuse overwrite. CPU<=3workers,BLAS threads1,
no GPU,installs,unrelated process operations or application edits.
Test conservative physics not actual-parameter leakage,truth invariance,
current SOC sensitivity,future-prefix invariance,expiry/outage bypass,
reference-style versus tail objective dispatch,physical continuity,
validation boundaries,selection fallback/guards and prediction-surplus clipping.
Independent NumPy audit hashes,actuator/supervisor/gates/selection/baseline
identity and paired contrasts locally. Re-run frozen v7 audit.
Always keep realtime_default_admitted=false; lab results are offline only.
