# V8 Statistical Appendix

## Units

One trained seed42,not independent sites. Two physical scenarios on the same
validation data are paired engineered conditions,not two randomized subjects.
All evaluation years were inspected in earlier rounds.
Exact daily SOC equality and hard bounds are deterministic constraints at
tolerance1e-5MWh,not estimates of field failure probability.

## Selection

Global retention uses nominal/combined90 validation P95 with frozen per-scenario
peak+25MW,P95+1MW and throughput/loss+10% guards,physical/recovery checks,
and mean P95 improvement>=1MW. No choice is made from evaluation outcomes.
Repeated use of calibration/validation data creates optimism risk.

## Contrasts and Intervals

32 paired conditions x three contrasts(two policies minus v7 and tail minus
reference) =96 comparisons. Circular7-day paired daily-peak blocks,
2000replicates,RNG20261026. Conditional on the fixed models and observed days.
No independent-hour tests,no confirmatory multiplicity-adjusted winner claim,
and no P95/maximum-load p-values or safety confidence limits.

Selected tail minus v7 supervised,fresh6:

| Scenario/Year | Mean Daily Peak Difference MW | Conditional95% Interval MW |
|---|---:|---|
|nominal2021|+55.39|[36.44,74.27]|
|nominal2022|+33.16|[17.94,48.90]|
|combined902021|-59.28|[-90.96,-31.46]|
|combined902022|-29.18|[-54.63,-6.58]|

These describe different condition-specific daily distributions. They do not
establish no change in the full-period extreme:combined90 maximum can worsen
even while mean daily maximum improves.

## Limits

No seed-population variance,new-grid generalization or independent-confirmatory
conclusion is available. Reused32 baseline trajectories are not new replications.
All numerical claims trace to the frozen per-hour outputs and independent verifier.
