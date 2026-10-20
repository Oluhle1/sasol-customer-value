"""Sprint 17 E12 — December-only probe exports for the transfer test.

Pre-registration: outputs/sprint17/proposal_transfer_probe.json.
Usage (venv python):
  /home/oluhle/.venvs/sasol-oct14/bin/python3 -u experiments/sprint17/export_probe.py pricefeat
  /home/oluhle/.venvs/sasol-oct14/bin/python3 -u experiments/sprint17/export_probe.py grumix
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "sprint17"))
sys.path.insert(0, str(ROOT / "experiments" / "sprint16"))

from evaluate_saved import CLASSES, OCT, rank4_combine, load_proba, load_rank4  # noqa: E402

OUT17 = ROOT / "outputs" / "sprint17"
TEST = "2025-12-01"


def log(message):
    print(f"{time.strftime('%H:%M:%S')} {message}", flush=True)


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("pricefeat", "grumix"):
        raise SystemExit("usage: export_probe.py pricefeat|grumix")
    variant = sys.argv[1]
    started = time.time()
    panel = pickle.loads((OCT / "category_dynamics_panel.pkl").read_bytes())
    frame = panel["frames"][TEST]
    ids = frame.index.astype(str)
    n = len(ids)
    vR = np.array(json.loads((ROOT / "outputs" / "sprint16" / "pool_r4full_vector.json").read_text())["vR"], dtype=float)
    p17 = load_proba("category_dynamics", TEST, n)

    if variant == "pricefeat":
        import fit_price_rank4 as fm
        p4 = fm.fit_rank(panel, TEST, extra=True)
    else:
        import fit_gru_rank4 as fm
        p4_base = load_rank4(TEST, n)
        if p4_base is None:
            raise RuntimeError("production rank4 cache missing for December")
        p4 = 0.5 * p4_base + 0.5 * fm.fit_gru(panel, TEST)

    prod = rank4_combine(p17, p4, "full")
    pred = np.asarray(CLASSES)[np.argmax(prod * vR, axis=1)]

    source = pd.read_csv(ROOT / "submissions" / "sasol_sprint16_r4full_vR.csv", dtype={"ID": str})
    if list(source["ID"].astype(str)) != list(ids):
        raise RuntimeError("December order does not match the source submission")
    out = source.copy()
    out["Opportunity"] = pred
    for col in ("CLV_fuel", "CLV_nonfuel"):
        if not np.array_equal(out[col].to_numpy(), source[col].to_numpy()):
            raise RuntimeError(f"{col} changed during export")
    path = ROOT / "submissions" / f"sasol_sprint17_{variant}_probe.csv"
    out.to_csv(path, index=False)
    changes = int((out["Opportunity"].to_numpy() != source["Opportunity"].to_numpy()).sum())
    audit = {
        "variant": variant,
        "proposal": "outputs/sprint17/proposal_transfer_probe.json",
        "source": "submissions/sasol_sprint16_r4full_vR.csv",
        "file": path.name,
        "rows": int(len(out)),
        "changes_vs_best": changes,
        "reference_public_score": 0.313217559,
        "promotion_rule": "delta >= +0.003 only",
        "seconds": round(time.time() - started, 1),
    }
    (OUT17 / f"{variant}_probe_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    log(f"exported {path.name}: {changes} label changes, {audit['seconds']}s")


if __name__ == "__main__":
    main()
