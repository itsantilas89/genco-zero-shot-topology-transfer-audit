"""Reproduction gate (README section 4).

Evaluates settings.yaml:gate.ckpt_id on its own grid with its own normaliser stats (variant A)
and compares with the metrics.csv shipped with the checkpoint. Exit code 1 if the numbers are
outside the tolerance, so run_all.sh stops before the expensive matrix.
"""
import json
import subprocess
import sys

import pandas as pd

from common import MANIFEST_PATH, RESULTS_DIR, ROOT, load_settings, python_exe, run_name
from postprocess import PER_RUN_DIR


def main() -> int:
    g = load_settings()["gate"]
    rn = run_name(g["ckpt_id"], g["target"], "A")
    (PER_RUN_DIR / f"{rn}.json").unlink(missing_ok=True)
    cmd = [python_exe(), str(ROOT / "scripts" / "run_matrix.py"), "--targets", g["target"],
           "--ckpts", g["ckpt_id"], "--variants", "A", "--keep-runs"] + sys.argv[1:]
    if subprocess.run(cmd, cwd=ROOT).returncode != 0 or not (PER_RUN_DIR / f"{rn}.json").exists():
        print("GATE FAILED: the evaluation run itself failed (see results/failures.csv)")
        return 1

    got = json.loads((PER_RUN_DIR / f"{rn}.json").read_text())
    man = pd.read_csv(MANIFEST_PATH, keep_default_na=False).set_index("ckpt_id").loc[g["ckpt_id"]]
    tol = g["rel_tolerance"]
    checks = []
    for label, key, expected_cfg, shipped in [
            ("active residual (MW)", "genco_active_res_MW", g["expected_active_res_MW"], man["id_active_res_MW"]),
            ("reactive residual (MVar)", "genco_reactive_res_MVar", g["expected_reactive_res_MVar"], man["id_reactive_res_MVar"])]:
        shipped = float(shipped)
        value = float(got[key])
        ok = abs(value - shipped) <= tol * shipped
        checks.append({"metric": label, "measured": value, "shipped_metrics_csv": shipped,
                       "readme_expected": expected_cfg, "within_tolerance": ok})
        print(f"{label:26s} measured {value:.5f}  shipped {shipped:.5f}  "
              f"({abs(value - shipped) / shipped:+.0%} off, tolerance {tol:.0%})  {'OK' if ok else 'FAIL'}")
    passed = all(c["within_tolerance"] for c in checks)
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "repro_gate.json").write_text(json.dumps(
        {"ckpt_id": g["ckpt_id"], "target": g["target"], "passed": passed,
         "n_test_scenarios": got.get("n_test_scenarios"), "checks": checks}, indent=2))
    print("GATE PASSED" if passed else "GATE FAILED: do not continue until this reproduces")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
