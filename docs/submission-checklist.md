# Submission Checklist

The acceptance basis for this checklist is the official competition requirements
and technical-report outline summarized in
[competition-alignment.md](competition-alignment.md). The project is a self-proposed
Technology Innovation Group entry, but the submission rules are official.

## Track and scope

- [x] Primary track selected: Technology Innovation Group.
- [x] Primary direction selected: new-energy consumption and grid-connection intelligent control.
- [x] Application scenario, method, evidence, limitations, and expected-benefit boundary are stated.
- [x] No claim of confirmed faults, real dispatch, realized savings, or deployed performance.

## Technical report

- [x] Repository-grounded technical report source exists at `docs/technical-report-draft.md`.
- [x] Final technical-report PDF exported to `outputs/report/gridpulse-technical-report-v0.1.pdf`.
- [x] Report follows the official structure: background, solution, experiments, innovation/application, conclusion, references, and appendix.
- [x] Citation includes OPSD DOI and required attribution.
- [x] No unverified savings, carbon reduction, user, deployment, or fault claims appear.
- [x] One-step causal backtest and theoretical storage scenario boundaries are visible.
- [x] Automated anonymous-material scan passes for report/PPT text, metadata, PPT notes, links, and source paths; final human review remains required.
- [x] Submission builder excludes `.git`, `.github`, local tooling files, `CITATION.cff`, and GitHub-profile links from the anonymous package.

## Defense PPT

- [x] Editable PPT exists at `outputs/presentation/gridpulse-defense-v9.pptx`.
- [x] Export the checked v9 PPT to `outputs/presentation/gridpulse-defense-v9.pdf`; the official submission format is PDF.
- [x] Automated anonymous-material scan passes for the PPT/PDF; final human review remains required.
- [x] Check chart and table values against their model, scenario, 2021/2022 validation, development, and optimality artifacts; review the rendered deck.
- [ ] Rehearse the project narrative and dashboard demonstration.

## Video

- [ ] Record the Streamlit workflow using `docs/demo-script.md`.
- [ ] Keep duration between 3 and 5 minutes and file size at or below 300 MB.
- [ ] Show the local application, risk evidence, limitations, and reproducibility path.
- [ ] Do not show local account names, school identifiers, browser tabs, or unrelated files.

## Other materials link

- [x] Include source code, reproducibility instructions, report source, deck, and data-acquisition instructions.
- [x] Build the local anonymous submission folder with `python scripts/build_submission_package.py --team-id <official-team-id>`.
- [x] Create a validated ZIP archive with `python scripts/archive_submission_package.py --package submission/<official-team-id>-AIC-AI能源-GridPulse`.
- [ ] Place the project deliverables in the chosen submission channel.
- [ ] Use the official team-number naming convention and a permanent shared-drive link for code, model, system prototype, and supplementary materials.

## Final preflight

- [x] Verify the delivered snapshot with `python scripts/final_preflight.py`, including frozen hashes, trajectory audits, certificates, and dashboard interactions.
- [x] Verify dependencies, tests, audits, and dashboard startup in a fresh isolated Windows environment.
- [x] Synchronize the supplied results with the report and PPT, including unfavorable results and costs.

Full source download and retraining remain separate reproduction work. Follow `docs/reproducibility.md` in a separate project copy to preserve the delivered frozen models and calibration; the original protocol hashes must continue to refer to their original artifacts.
