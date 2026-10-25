"""Write a random results.csv / grid_scales.csv with the real schema, to exercise
analyses.py and figures.py without running the matrix.

    AUDIT_RESULTS_DIR=/tmp/synth AUDIT_FIGURES_DIR=/tmp/synth/fig python scripts/synthetic_results.py
The numbers are meaningless; never point this at the real results/ folder.
"""
import os
import sys

import numpy as np
import pandas as pd

from common import MANIFEST_PATH, RESULTS_DIR, ROOT, load_settings, run_name
from run_matrix import variants_for

if RESULTS_DIR == ROOT / "results":
    sys.exit("refusing to overwrite the real results/ folder: set AUDIT_RESULTS_DIR")

rng = np.random.default_rng(1)
man = pd.read_csv(MANIFEST_PATH, keep_default_na=False)
targets = list(load_settings()["targets"])
scales = {"case14_ieee": 17.0, "case30_ieee": 8.1, "case57_ieee": 9.0, "case118_ieee": 12.0,
          "case500_goc": 20.0, "case2000_goc": 25.0, "case10000_goc": 30.0}
rows = []
for r in man.itertuples():
    for t in targets:
        for v in variants_for(r, t):
            diag = r.source_grid == t
            rel = 10 ** rng.uniform(-4.5, -3.5) if diag else 10 ** rng.uniform(-1.5, 0.3)
            dc = 10 ** rng.uniform(-0.5, 0.5)
            rows.append({
                "run_name": run_name(r.ckpt_id, t, v), "ckpt_id": r.ckpt_id,
                "source_grid": r.source_grid, "size": r.size, "seed": r.seed, "target": t,
                "variant": v, "is_diagonal": diag, "target_mean_S_MVA": scales[t],
                "rel_active_res": rel * (3 if v == "B" and not diag else 1),
                "genco_active_res_MW": rel * scales[t], "dc_active_res_MW": dc,
                "genco_over_dc": rel * scales[t] / dc, "vm_mae_all": rel / 5,
                "vm_violation_rate": rng.uniform(0, .3), "vm_true_violation_rate": .03,
                "vm_cal_slope": rng.uniform(-.2, 1.1), "va_cal_slope": rng.uniform(-.2, 1.1),
                "vm_cal_r2": rng.uniform(0, 1), "va_cal_r2": rng.uniform(0, 1)})
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(RESULTS_DIR / "results.csv", index=False)
pd.DataFrame({"grid": list(scales), "mean_bus_apparent_power_MVA": list(scales.values())}
             ).to_csv(RESULTS_DIR / "grid_scales.csv", index=False)
print(f"wrote {len(rows)} synthetic runs to {RESULTS_DIR}")
