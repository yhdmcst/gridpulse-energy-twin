"""Verify an isolated Python installation without retraining frozen artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python",required=True,type=Path)
    args = parser.parse_args()
    exe = args.python.resolve()
    code = "import sys,json,pathlib; p=pathlib.Path(sys.prefix)/'pyvenv.cfg'; print(json.dumps({'python':sys.version.split()[0],'isolated_venv':sys.prefix!=sys.base_prefix,'system_site_packages': 'include-system-site-packages = true' in p.read_text().lower() if p.exists() else True}))"
    probe = subprocess.run([str(exe),"-c",code],capture_output=True,text=True,check=True)
    environment = json.loads(probe.stdout)
    if not environment["isolated_venv"] or environment["system_site_packages"]:
        raise ValueError("An isolated venv without system site packages is required.")
    checks = []
    commands = [("dependency_consistency",["-m","pip","check"]),
        ("unit_tests",["-m","pytest","-q"]),
        ("optimality_certificates",["scripts/certify_optimality.py","--verify-only"]),
        ("dashboard",["scripts/check_app.py"]),
        ("development_audit",["scripts/audit_development.py"]),
        ("later_year_audit",["scripts/audit_later_year.py"])]
    for name,command in commands:
        result = subprocess.run([str(exe),"-B",*command],cwd=ROOT,capture_output=True,text=True,
            encoding="utf-8",errors="replace",timeout=180,
            env={**os.environ,"PYTHONIOENCODING":"utf-8","PYTHONDONTWRITEBYTECODE":"1"})
        # Avoid exporting local paths and testing temp paths into submission.
        checks.append({"name":name,"passed":result.returncode == 0,"exit_code":result.returncode})
        print(name,result.returncode,flush=True)
        if result.returncode:
            print((result.stdout+result.stderr)[-2500:],flush=True)
    lock = subprocess.run([str(exe),"-m","pip","freeze"],capture_output=True,text=True,check=True).stdout
    (ROOT/"outputs/development/clean_environment_requirements.txt").write_text(lock,encoding="utf-8")
    receipt = {"schema_version":"gridpulse-clean-environment.v1","platform":"Windows x64",
        "environment":environment,"checks":checks,"passed":all(c["passed"] for c in checks),
        "requirements_sha256":hashlib.sha256((ROOT/"requirements.txt").read_bytes()).hexdigest(),
        "boundary":"Fresh isolated dependency install and existing-artifact tests. Full source download/retraining and other operating systems were not exercised."}
    (ROOT/"outputs/development/clean_environment_verification.json").write_text(json.dumps(receipt,indent=2),encoding="utf-8")
    print(json.dumps(receipt,indent=2))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
