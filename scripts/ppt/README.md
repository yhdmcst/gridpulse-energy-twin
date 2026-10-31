# Local presentation build dependency

Run `npm run ppt:setup` once on a machine with the required local presentation runtime. This places `@oai/artifact-tool` in this repository's `node_modules` for local development.

`@oai/artifact-tool` is a private presentation dependency unavailable from the public npm registry. It is deliberately installed without saving an absolute machine path to `package.json`. The repository therefore does not attempt to publish or vendor the package.

Use `npm run ppt:build` to build the deck and `npm run ppt:finalize` to run the local finalization flow. Before finalizing in PowerShell, set `GRIDPULSE_PYTHON` to an absolute Python path and `GRIDPULSE_PRESENTATION_SKILL_DIR` to the local presentations-skill directory. The tracked build sources are `build_gridpulse_deck.mjs` and `finalize_gridpulse_deck.mjs`; generated local build artifacts remain ignored by Git.

The current finalizer also requires `RUNTIME_NODE_MODULES` to point at the bundled runtime's Node package directory for its import check. The current revision exports a fifteen-slide deck, including native editable daily-energy, peak-response, 2021/2022 frozen-validation, optimality and controller-development tables, to `outputs/presentation/gridpulse-defense-v9.pptx`. This private authoring dependency is unnecessary for running the dashboard or reproducing the numerical experiments.
