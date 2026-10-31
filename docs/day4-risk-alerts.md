# Candidate-Risk Screening

## Scope and terminology

The alert layer screens for candidate operational risks. It does not diagnose equipment faults, because the OPSD release has no verified fault/event labels for this purpose.

## Three alert rules

1. **Peak-load risk**: predicted net load is at or above the validation-period 95th percentile.
2. **Solar-drop risk**: the observed hourly solar change is at or below the validation-period 5th percentile.
3. **Sustained-deviation risk**: the absolute forecast residual exceeds the validation-period 97.5th percentile for at least three consecutive hours with the same sign. The flag first appears at the third observation (two hours after onset), without retrospectively flagging the first two points. Missing hours reset the run.

Every test alert retains the trigger value, threshold, risk type, and evidence string. Thresholds are calibrated only on the frozen validation period (`2019-01-12 20:00 UTC` through `2019-08-09 20:00 UTC`) and then applied unchanged to the test period.

The alert export also carries the validation-calibrated prediction upper bound, an interval-based peak watch flag, a transparent risk score, and a review priority. The interval watch does not silently convert into a confirmed event: it is an additional human-review signal when the point prediction is below the peak threshold but the empirical upper bound reaches it.

## Calibrated thresholds

- Peak predicted net load: `55,406.34 MW`
- Solar hourly drop: `-4,243.00 MW/h`
- Absolute residual: `2,936.54 MW`
- Sustained duration: `3 hours`

The test period contains 972 rows with at least one candidate-risk flag. This is a screening count, not a fault count or accuracy claim.

## Synthetic stress test

Because real labels are unavailable, the implementation injects 25 deterministic three-hour stress events per category into a copy of the test frame. The seed is `20260929`; injected magnitudes are recorded in `outputs/alerts/alert_evaluation.json`.

| Injected category | Point detections | Injected points | Detection rate |
|---|---:|---:|---:|
| Peak load | 75 | 75 | 100% |
| Solar drop | 25 | 25 | 100% |
| Sustained deviation | 25 | 75 | 33.33% |

All 25 sustained-deviation events are detected at their third hour, with a two-hour confirmation delay. Event detection (25/25) and point detection (25/75) measure different things. Peak detection is 75/75 points; solar-drop detection is 25/25 injected drop points.

These rates only verify that the rule mechanics respond to the chosen artificial scenarios. They are not historical detection performance and must not be presented as real fault-detection accuracy.

## Outputs

- Implementation: `src/alerts.py`
- Runner: `scripts/run_alerts.py`
- Frozen thresholds and stress results: `outputs/alerts/alert_evaluation.json`
- Auditable test alert rows: `outputs/alerts/test_candidate_risks.csv`
- Human-review queue, including interval-only peak watches: `outputs/alerts/test_review_queue.csv`

## Known limitations

The rules are percentile-based and region-specific. The solar-drop rule uses actual solar observations, which are suitable for retrospective evaluation but not automatically available at a real forecast issuance time. A production version needs an explicit information-availability design, weather/renewable forecasts, alert deduplication, and expert review.
