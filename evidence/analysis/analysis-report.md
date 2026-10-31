# V8 Evidence Assessment

Date:2026-10-28. Question:does full-window conservative planning reduce v7
posthoc-recovery cost without losing physical/daily recovery or performance?

## Evidence

Fixed seed42,already inspected2021/2022,363days/8712h per evaluation trace.
Conservative875MW/.85 model and virtual reference replace1250MW/.92 planning,
but true plant values remain separated from the decision function.
Both reference-objective and tail-objective pipelines are tested.
Six validation traces use the original common validation period; retained
envelope_tail before reading this round's evaluation outcomes.

Independent CSV/hash audit passed204 artifacts and114 total trajectories:
6validation,12references,64new evaluation,32byte-reused v7 supervised baselines.
All96 declared pairwise interval sets recomputed; no failed experimental jobs.

## Supported Findings

- Both new methods satisfy recovery in60/60 in-envelope evaluation scenarios.
- Retained tail policy satisfies30/30 and records0 natural LP failures and0
  posthoc supervisor interventions in those conditions.
- Expiry/outage handling remains active; zero natural LP failures does not imply
  zero fallback during scheduled service/data faults.
- Validation retention is not a performance guarantee:nominal fresh6 evaluation
  maxima worsen28.57/48.16MW for2021/2022 relative to v7 supervised;2022 P95
  worsens49.40MW. These exceed old descriptive protection budgets.
- Some mismatch cases improve maximum load substantially,while combined90
  peaks can worsen54.88/271.48MW. Costs are conditional,not universally better.
- Outside-envelope tail recovery still fails642.96/589.59MWh. Improvement in
  a counterexample does not extend the declared physical guarantee.

## Interpretation and Decision

Planning the conservative recovery path avoids many v7 infeasible reference
requests/posthoc changes. However,conservative power limits reduce how much
energy is available for peak shaving and alter the whole virtual reference.
The ablation does not identify a single constraint's independent causal effect.

Retain the validation-selected experimental label,report all evaluation failures,
and do not promote it to the default realtime application. No evaluation reselection.
Further enlarged sweeps on these same years risk adapting to already-known outcomes.
