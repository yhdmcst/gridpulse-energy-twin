# Fusion v4: Validation-Selected Tail-Aware Six-Hour Control

Frozen before v4 simulations on 2026-10-26. V1-v3 and all source years have
already been examined. Exploratory engineering, not an untouched confirmation.

## Question

Can changing the rolling objective and advisory reserve alleviate v3 P95
regression while preserving terminal energy, realized peak and cycling budgets?
No model retraining, new data, GPU allocation, price or deployment claims.

## Frozen Inputs and Shared Physics

Use v3 raw six-horizon HGB and data MLP seeds42/43/44. Do not select a seed.
Use exactly v3 complete validation/test/2021/2022 UTC hours, calibrated forecast
bands, uncertainty scales and q95/q25/q99 gates. Add one fixed tail threshold:
q90 of the common validation actual h1 net load. No later-year quantiles.

Same physical model: capacity30000MWh, initial/ceiling/terminal15000MWh,
minimum4500MWh, charge/discharge1250MW, per-direction efficiency.92.
No free reset, hidden recharge, actual-load optimization or omission of failures.
Same idealized observation availability and six-horizon issuance policy as v3.

## Seven Controllers

Legacy is the unchanged v3 six-hour balanced controller.
Six new controls use an existing two-stage LP pattern with SciPy HiGHS:

1. peak_dynamic: minimize window peak, then throughput at that optimal peak.
2. peak_physical: same objective, reserve at hard physical minimum.
3. tail_dynamic_0: minimize peak, then mean positive forecast excess above
   the validation q90 threshold plus .001 times throughput, peak cap z_min.
4. tail_dynamic_250: same, peak cap z_min+250MW.
5. tail_physical_0: tail objective, hard-minimum reserve, peak cap z_min.
6. tail_physical_250: tail objective, hard-minimum reserve, cap z_min+250MW.

Dynamic reserve, q99 release and forecast-upper discharge gates remain as in v3.
Physical reserve removes the advisory floor only, never hard SOC or daily
reachability. Recovery charging remains available at every hour and fully counted.
Reachability lower/upper SOC bounds use actual modeled powers and efficiency.
Both LP stages use point forecasts; calibrated bands still determine reserve/gates.
Allow negative peak variable; do not clamp valid negative net load to zero.
Remove simultaneous LP actions preserving SOC, reducing grid load and throughput,
then independently check the complete planned trajectory and executed trace.

The tail term is a threshold-exceedance PROXY, not an optimizer of empirical P95,
not CVaR with a fitted quantile, and not a guaranteed out-of-sample risk bound.
The 250MW cap is a declared local forecast tradeoff, not a realized-peak guarantee.

## Validation-Only Selection

Run all seven controls for all four predictors on common validation hours:
28 validation traces. Check full validation lower/upper label bounds.

Select ONE global policy using only the three data MLP seeds (HGB is a transfer
check and must not influence selection). For every seed, require:
- all physical and terminal audits pass;
- realized maximum <= its legacy maximum+25MW;
- throughput <=1.10 times legacy throughput+1e-6MWh;
- conversion loss <=1.10 times legacy loss+1e-6MWh;
- P95 <= legacy P95+1MW.

Also require mean P95 improvement over the three seeds of at least1MW.
Among eligible new controls choose minimum mean P95; break ties by mean
conversion loss then lexicographic policy ID. If none pass, select legacy.
Never force a new policy to win or expand the grid after seeing results.
Save selection.json with validation result hashes, choice, guard outcomes,
UTC freeze time and SHA256 BEFORE reading/simulating evaluation-period inputs.
Document the in-sample calibration/selection reuse and resulting optimism risk.

## Locked Evaluation and Stress

Evaluate ALL seven controls x four predictors x three periods =84 nominal
traces, including the frozen legacy traces reused byte-for-byte from v3.
The six new controls require72 simulations. Keep selected status fixed even
if another ablation looks better on a known evaluation period.

Report per seed and predictor: no-storage and controlled max/P95, mean daily
peak, mean excess above the FIXED validation tail threshold, throughput,
conversion loss, recovery charging, actual high-load charging and terminal errors.
Pair each new policy with its same-predictor legacy. Seven-day circular
paired blocks,2000 replicates,RNG20261026 for mean daily peak and mean daily
tail-excess differences; fixed models and observed days only.
P95 differences are descriptive, not a percentile of independently sampled hours.

Fixed stress probe data_seed42 with the selected policy. For2021/2022, compare
continuous nominal-action replay and parameter-aware replanning under875MW
both powers or efficiency.85. Eight traces. Parameters are perfectly known at
episode start. No health sensing, sudden unobserved fault or field-safety claim.
If legacy is selected, still keep it and report selection failure transparently.

## Provenance, Budget and Gates

New outputs/fusion_v4, refuse overwrite. At most3 CPU workers, one numerical
thread each; no installs/driver updates/unrelated process interference.
Record all source/protocol/input hashes, samples, thresholds, validation guard
failures, selection receipt and all per-hour traces. Preserve solver failures.
Preflight: truth invariance, split bounds, selection independence from HGB,
fallback when guards fail, unchanged daily reachability, SOC-preserving action
cancellation, negative-net support, forecast peak-cap checks, and strict input
validation. Independently audit delivered CSVs/selection/contrasts locally;
verify v1-v3 frozen integrity again before reporting.
