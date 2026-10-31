import { existsSync } from "node:fs";
import { spawnSync } from "node:child_process";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const configuredSource = process.env.GRIDPULSE_ARTIFACT_TOOL_SOURCE;
const bundledSource = path.join(
  process.env.USERPROFILE ?? process.env.HOME ?? "",
  ".cache", "codex-runtimes", "codex-primary-runtime",
  "dependencies", "node", "node_modules", "@oai", "artifact-tool",
);
const source = configuredSource ?? bundledSource;

if (!existsSync(path.join(source, "package.json"))) {
  throw new Error(
    "Cannot find the locally supplied @oai/artifact-tool package. Set GRIDPULSE_ARTIFACT_TOOL_SOURCE to its directory.",
  );
}

const npm = process.platform === "win32" ? "npm.cmd" : "npm";
const result = spawnSync(
  npm,
  ["install", "--no-save", "--ignore-scripts", `file:${source}`],
  { cwd: repoRoot, stdio: "inherit" },
);
if (result.status !== 0) process.exit(result.status ?? 1);
console.log("Installed @oai/artifact-tool into this repository's node_modules.");
