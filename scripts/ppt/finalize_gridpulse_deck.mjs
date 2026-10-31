import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const workspaceDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const skillDir = process.env.GRIDPULSE_PRESENTATION_SKILL_DIR;
if (!skillDir || !path.isAbsolute(skillDir)) {
  throw new Error(
    "Set GRIDPULSE_PRESENTATION_SKILL_DIR to the absolute presentations skill directory before finalizing the deck.",
  );
}
const runtimePython = process.env.GRIDPULSE_PYTHON;
if (!runtimePython || !path.isAbsolute(runtimePython)) {
  throw new Error(
    "Set GRIDPULSE_PYTHON to the absolute path of a Python executable before finalizing the deck.",
  );
}
const { finalizePresentation } = await import(pathToFileURL(path.join(skillDir, "container_tools", "artifact_tool_utils.mjs")).href);
const stagingDir = path.join(workspaceDir, ".codex-finalizer");
const candidatePath = path.join(workspaceDir, "tmp", "ppt-build", "gridpulse-defense-v9-draft.pptx");
const finalPath = path.join(workspaceDir, "outputs", "presentation", "gridpulse-defense-v9.pptx");
await fs.mkdir(stagingDir, { recursive: true });
await fs.mkdir(path.dirname(finalPath), { recursive: true });
const result = await finalizePresentation({
  workspaceDir,
  candidatePath,
  finalPath,
  pythonExecutable: runtimePython,
  integrityValidatorPath: path.join(skillDir, "container_tools", "inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(skillDir, "container_tools", "inspect_presentation_layout_geometry.py"),
  layoutArgs: ["--expected-slide-size-emu", "12192000,6858000", "--validate-bullet-geometry", "--validate-heading-fit", "--require-native-table-slide", "8", "--require-native-table-slide", "9", "--require-native-table-slide", "10", "--require-native-table-slide", "11", "--require-native-table-slide", "12", "--require-native-table-slide", "13", "--require-native-table-slide", "14"],
  explicitTotalSlideCount: 15,
  requiredNativeTableOwnerSlides: [8, 9, 10, 11, 12, 13, 14],
  requiredNativeChartOwnerSlides: [5, 7],
  materializeLiteralChartWorkbooks: true,
  fontPolicy: { basis: "design", families: ["Microsoft YaHei"] },
  verifyArtifactToolImport: true,
  receiptPath: path.join(stagingDir, "gridpulse-defense-v9.validation.json"),
});
console.log(JSON.stringify(result, null, 2));
