# Figure Contracts

Destination: AIC AI+Energy technology-innovation technical report, not a Nature
journal submission. Official competition typography and structure take priority.
Backend: existing Python/Matplotlib workflow, saved workspace preference.
No private plotting collection, API image generation, fabricated data or demo values.
Reuse: structural adaptation of project figures; user skill supplies QA framework.
Standalone plotting width: 172-183 mm; report placement width:154 mm following
the reference template margins. Editable SVG/PDF and300dpi PNG; glyph floor5pt,
also checked after report scaling (minimum about6.40pt).
Every multipanel plot: measured final axes, panel letters, 1.5 pt alignment gate.
Every final figure: source validation, PDF glyph audit, PDF collision audit.

| Figure | Question and bounded conclusion | Distinct panel roles | Replication/uncertainty |
| --- | --- | --- | --- |
| 1 Architecture | Which information enters the offline loop? Explicit simulator/control boundary. | Single schematic with feedback and audit path. | Not a quantitative result. |
| 2 Timeline | Is a delayed packet still aligned to its original target lead? | Time split plus original-lead mapping. | Logical, not duration-proportional. |
| 3 Fallback | What happens before/after a primary solver call? Invalid input and service failure are distinct. | Single workflow; no claim about physical power outage. | Declared exact SOC and charging-availability assumptions. |
| 4 Prediction | Is prediction gain due to the network or consistency regularizer? | a matched HGB comparison; b seed-paired accuracy effect; c accounting residual. | All fixed seeds42/43/44, raw points; HGB single model; no p-values. |
| 5 Dispatch | Does a longer horizon improve every operational objective? No. | a horizon maximum delta; b horizon P95 delta; c guarded-tail P95 effect. | All 3 paired seeds per period; means plus raw paired differences. |
| 6 Robustness | Do old packets and wrong parameters preserve nominal conclusions? Not in general. | a delay-specific effect; b fixed-request terminal failure. | 3 fixed seeds for nominal delay; fixed-seed mismatch probes; no formal CI. |
| 7 Feedback | Is exact SOC enough, and what does supervision cost? | a paired trajectory recovery; b nominal peak/P95 cost. | 24 healthy pairs; 30 including faults stated in caption; seed42 only. |
| 8 Envelope | Does planning consistency also guarantee performance dominance? No. | a P95 gains/losses; b maximum regression. | Fixed seed42, both years and both stated fresh6 scenarios. |
| 9 Full conditions | Does candidate satisfy recovery and old performance budgets throughout? 30/30 versus11/30. | a maximum effect; b P95 effect; c five budget states. | All30 stated conditions, correlated scenarios, not independent trials. |
| 10 Fault replay | Does the implementation bypass optimization during declared faults? | a energy/floor; b actual actions; c call/fallback flags. | First scheduled2022 combined-fault UTC day, not performance-selected. |

Data scope: all requested periods/conditions retained in source tables. Narrative
figures explicitly label any subset (fresh6 nominal/combined90 or first fault day);
the full 30-condition view and source data preserve excluded views. No resampling,
new significance testing, smoothing or hidden model reselection.

Blockers: collision FAIL or alignment FAIL prevents final export acceptance.
Automated PASS cannot establish complete visual quality or scientific validity.
