"""Post-hoc offline bounds; never update the frozen forecasting/control protocol."""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.optimality import (StorageModel, analytic_certificate, offline_benchmark,
    verify_benchmark, exact_power_bound_witness, verify_exact_witness)
from src.numerics import same_numeric_evidence

CASES = {
    "independent_2021":"outputs/validation/independent_2021_dynamic_peak_response_timeseries.csv",
    "exploratory_test":"outputs/scenarios/peak_test_dynamic_peak_response_timeseries.csv",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_inputs(path):
    with path.open(encoding="utf-8",newline="") as stream:
        rows = list(csv.DictReader(stream))
    return [r["timestamp"] for r in rows],[r["net_load_mw"] for r in rows]


def verify(root=ROOT):
    report = json.loads((root/"outputs/optimality/certificates.json").read_text(encoding="utf-8"))
    if report["schema_version"] != "gridpulse-optimality.v1" or set(report["cases"]) != set(CASES):
        return False
    for name,source in CASES.items():
        case = report["cases"][name]
        if case["input"] != source or digest(root/source) != case["input_sha256"]:
            return False
        frame = pd.read_csv(root/source)
        oracle_path = root/f"outputs/optimality/{name}_oracle.csv"
        if digest(oracle_path) != case["oracle_sha256"]:
            return False
        trace = pd.read_csv(oracle_path)
        oracle = case["offline_benchmark"]
        if oracle["model"] != asdict(StorageModel()):
            return False
        if not verify_benchmark(frame[["timestamp","net_load_mw"]],trace,oracle):
            return False
        if not verify_exact_witness(*exact_inputs(root/source),StorageModel(**oracle["model"]),case["exact_rational_certificate"]):
            return False
        exact = case["exact_rational_certificate"]
        if (abs(float(Fraction(exact["attained_peak_mw"]))-oracle["period_optimum_peak_mw"]) > 1e-5
                or abs(float(Fraction(exact["attained_throughput_mwh"]))-oracle["minimum_throughput_at_optimum_peak_mwh"]) > 1e-3):
            return False
        analytic = analytic_certificate(frame,StorageModel(**oracle["model"]))
        if not same_numeric_evidence(analytic, case["analytic_certificate"]) or not analytic["audit"]["passed"]:
            return False
        if abs(case["controller_peak_gap_mw"]-(analytic["achieved_peak_mw"]-oracle["period_optimum_peak_mw"])) > 1e-5:
            return False
        excess = (frame.charge_mw+frame.discharge_mw).sum()-oracle["minimum_throughput_at_optimum_peak_mwh"]
        if abs(case["controller_throughput_excess_over_offline_mwh"]-excess) > 1e-3:
            return False
    return True


def generate():
    folder = ROOT/"outputs/optimality"
    folder.mkdir(parents=True,exist_ok=True)
    report = {"schema_version":"gridpulse-optimality.v1",
        "status":"Post-hoc finite-instance offline certificates; frozen policy and validation unchanged.",
        "information":"Oracle sees all actual loads. Controller sees only its originally frozen forecast information.",
        "objective_priority":["Minimize full-period maximum","Then minimize grid-side throughput at that maximum"],
        "scope":"Complete independent UTC days; 1-hour steps; fixed physical model. Advisory reserve and forecast gates excluded from oracle.",
        "numerics":"LP primal-dual checks use explicit floating tolerances. Separate rational lower-bound/attainment certificates use exact decimal input strings and zero tolerance.",
        "cases":{}}
    model = StorageModel()
    for name,source in CASES.items():
        frame = pd.read_csv(ROOT/source)
        trace,oracle = offline_benchmark(frame[["timestamp","net_load_mw"]],model)
        path = folder/f"{name}_oracle.csv"
        trace.to_csv(path,index=False)
        analytic = analytic_certificate(frame,model)
        report["cases"][name] = {"input":source,"input_sha256":digest(ROOT/source),
            "oracle_sha256":digest(path),"analytic_certificate":analytic,"offline_benchmark":oracle,
            "exact_rational_certificate":exact_power_bound_witness(*exact_inputs(ROOT/source),model),
            "controller_peak_gap_mw":analytic["achieved_peak_mw"]-oracle["period_optimum_peak_mw"],
            "controller_throughput_excess_over_offline_mwh":float((frame.charge_mw+frame.discharge_mw).sum())-oracle["minimum_throughput_at_optimum_peak_mwh"]}
    (folder/"certificates.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    if not verify():
        raise RuntimeError("Independent certificate verification failed.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only",action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        passed = verify()
        print(json.dumps({"certificate_verification_passed":passed}))
        return 0 if passed else 1
    report = generate()
    print(json.dumps({name:{"global_peak_mw":case["offline_benchmark"]["period_optimum_peak_mw"],
        "controller_peak_gap_mw":case["controller_peak_gap_mw"],
        "minimum_offline_throughput_mwh":case["offline_benchmark"]["minimum_throughput_at_optimum_peak_mwh"],
        "analytic_attainment":case["analytic_certificate"]["globally_optimal_within_tolerance"]}
        for name,case in report["cases"].items()},ensure_ascii=False,indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
