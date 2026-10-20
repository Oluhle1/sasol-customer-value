"""Sprint 17 E13 — assemble the Colab bundle for the TabPFN member.

Version-agnostic by design: the panel is exported as plain numpy arrays and
Python primitives (no pandas objects), so any Colab pandas/numpy combination
can load it. Copies only frozen artifacts; no fitting.
Outputs outputs/sprint17/colab_bundle.zip.
"""
import json
import pickle
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT17 = ROOT / "outputs" / "sprint17"
STAGE = Path("/tmp/opencode/sasol_colab_bundle")
ALL = ["2025-03-01", "2025-04-01", "2025-05-01", "2025-06-01",
       "2025-07-01", "2025-08-01", "2025-09-01", "2025-12-01"]


def log(message):
    print(f"{time.strftime('%H:%M:%S')} {message}", flush=True)


def main():
    if STAGE.exists():
        shutil.rmtree(STAGE)
    (STAGE / "probs").mkdir(parents=True)
    (STAGE / "rank4").mkdir(parents=True)

    panel = pickle.loads((ROOT / "outputs" / "oct10" / "category_dynamics_panel.pkl").read_bytes())
    plain = {"feature_cols": list(panel["feature_cols"]),
             "monthly_cutoffs": list(panel["monthly_cutoffs"]),
             "cutoffs": {}}
    for cutoff in panel["frames"].keys():
        frame = panel["frames"][cutoff]
        x = frame[panel["feature_cols"]].to_numpy(dtype=np.float32)
        ids = np.asarray(frame.index.astype(str).tolist(), dtype=str)
        lab = panel["labels"].get(cutoff)
        y = lab.reindex(frame.index).Opportunity.tolist() if lab is not None else None
        plain["cutoffs"][cutoff] = {"X": x, "ids": ids, "y": y}
        log(f"plain {cutoff}: X {x.shape}, labels {'yes' if y is not None else 'no'}")

    for cutoff in ALL:
        frame = panel["frames"][cutoff]
        n = len(frame)
        if cutoff in ("2025-03-01", "2025-04-01", "2025-05-01", "2025-07-01", "2025-08-01"):
            proba = np.load(ROOT / "outputs" / "top10_round" / "probs" / f"{cutoff}_category_dynamics.npz",
                            allow_pickle=False)["proba"]
        else:
            proba = np.asarray(np.load(ROOT / "outputs" / "oct10" / f"{cutoff}_category_dynamics.npy",
                                       allow_pickle=True), dtype=float)
        if proba.shape != (n, 17):
            raise RuntimeError(f"{cutoff}: proba {proba.shape} vs frame {n}")
        np.savez(STAGE / "probs" / f"{cutoff}.npz", proba=proba)

        src4 = ROOT / "outputs" / "sprint17" / "price_rank4" / f"{cutoff}_rank4_base.npz" \
            if cutoff == "2025-03-01" else ROOT / "outputs" / "top10_round" / "rank4" / f"{cutoff}_rank4.npz"
        p4 = np.load(src4, allow_pickle=False)["proba"]
        if p4.shape != (n, 4):
            raise RuntimeError(f"{cutoff}: rank4 {p4.shape} vs frame {n}")
        np.savez(STAGE / "rank4" / f"{cutoff}.npz", proba=p4)

    with (STAGE / "panel_plain.pkl").open("wb") as fh:
        pickle.dump(plain, fh, protocol=4)

    shutil.copy2(ROOT / "outputs" / "sprint16" / "pool_r4full_vector.json", STAGE / "pool_r4full_vector.json")
    shutil.copy2(ROOT / "submissions" / "sasol_sprint16_r4full_vR.csv", STAGE / "sasol_sprint16_r4full_vR.csv")
    shutil.copy2(ROOT / "experiments" / "sprint17" / "colab_tabpfn.py", STAGE / "colab_tabpfn.py")
    (STAGE / "README.txt").write_text(
        "Sasol sprint 17 E13 TabPFN member - Colab bundle (v2, version-agnostic panel)\n"
        "\n"
        "Run from Google Drive so results survive disconnects:\n"
        "1. Upload colab_bundle.zip to Drive (MyDrive/sasol/).\n"
        "2. Runtime > Change runtime type > T4 GPU.\n"
        "3. Cell: from google.colab import drive; drive.mount('/content/drive')\n"
        "4. Cell: !mkdir -p /content/drive/MyDrive/sasol && cd /content/drive/MyDrive/sasol && unzip -o -q colab_bundle.zip -d bundle\n"
        "5. Cell: !pip install -q tabpfn\n"
        "6. Cell: !cd /content/drive/MyDrive/sasol/bundle && python colab_tabpfn.py 2>&1 | tee /content/drive/MyDrive/sasol/tabpfn_run.log\n"
        "7. Results land on Drive: bundle/tabpfn_decision.json, bundle/tabpfn/*.npz, bundle/sasol_sprint17_tabpfnmix.csv (only if gates pass).\n"
        "   If the session dies, rerun cell 6: finished origins are cached and skipped.\n"
    )

    zip_path = OUT17 / "colab_bundle.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(STAGE.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(STAGE))
    size_mb = zip_path.stat().st_size / 1e6
    log(f"wrote {zip_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
