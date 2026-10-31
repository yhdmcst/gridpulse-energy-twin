# Anonymous Submission Review

Run the automated material check before packaging:

```powershell
python scripts/anonymous_preflight.py
```

It scans the final report PDF, defense PPTX/PDF, speaker notes, embedded PPTX XML,
external-link relationships, document metadata, and local paths in submitted source
scripts. The command prints its findings and saves a local verification receipt.

## Manual release gate

The scanner cannot see a presenter, browser tab, screen recording, or infer identity
from a visual image. Before handing materials to anonymous review:

- Do not include `.git`, `.github`, Git history, local IDE settings, caches, or logs.
- Do not include `CITATION.cff` or a GitHub-profile link in an anonymous submission
  package unless the organizer explicitly requests a repository identity.
- Verify the shared-drive URL, video filename, browser profile, and recording desktop
  do not contain names, school information, account handles, or local directories.
- Keep third-party attribution in the report and source notice; remove only identity
  metadata that is not required for evaluation.

The public GitHub repository may retain its normal authorship and citation metadata.
That public-repository policy is separate from the anonymous competition package.
