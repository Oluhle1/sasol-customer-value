"""Sprint 17 E13 - TabPFN rank4 member (run on Colab, GPU runtime).

Pre-registration: outputs/sprint17/proposal_tabpfn.json (frozen in the repo).
Self-contained and pandas-free: loads only panel_plain.pkl (plain numpy), the
probs/ and rank4/ npz caches, pool_r4full_vector.json, and the source CSV.
Exports the December submission with the csv module (CLV strings preserved).

Colab usage:
  !pip install -q tabpfn
  !python colab_tabpfn.py
Runtime: T4 GPU (Runtime > Change runtime type). CPU fallback works but is slow.
"""
import csv
import json
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

HERE = Path(__file__).resolve().parent
POOL = ["2025-03-01", "2025-04-01", "2025-05-01", "2025-07-01", "2025-08-01"]
EVAL = ["2025-06-01", "2025-09-01"]
TEST = "2025-12-01"
RANK = ["Inactivity", "Stable", "Fuel", "Other"]
FUEL_LABEL = "Existing-category growth: Fuel"
CAP = 10000
SCREEN_GATE = {"mean_gain_min": 0.004, "per_origin_min": -0.002}
EVAL_GATE = {"per_origin_min": 0.0015, "mean_gain_min": 0.004}
BASELINE_GUARD = {
    "2025-03-01": 0.4822916102639962,
    "2025-04-01": 0.5006451168766207,
    "2025-05-01": 0.5060866657571113,
    "2025-06-01": 0.5111014600824757,
    "2025-07-01": 0.5031954596858046,
    "2025-08-01": 0.5059436172953636,
    "2025-09-01": 0.5261783234452059,
}
CLASSES = [
    "Existing-category growth: Beverages",
    "Existing-category growth: Food & beverage",
    "Existing-category growth: Fuel",
    "Existing-category growth: Other",
    "Inactivity",
    "New category adoption: Beverages",
    "New category adoption: Braai & Ice",
    "New category adoption: Breads & rolls",
    "New category adoption: Confectionary",
    "New category adoption: Food & beverage",
    "New category adoption: Groceries",
    "New category adoption: Ice-cream",
    "New category adoption: Lubricants",
    "New category adoption: Other",
    "New category adoption: Snacking",
    "New category adoption: Tobacco",
    "Stable",
]
IDX = {c: i for i, c in enumerate(CLASSES)}
BIG = [IDX["Inactivity"], IDX["Stable"], IDX[FUEL_LABEL]]
OTHER = [i for i in range(17) if i not in BIG]
RANK_CODE = {name: i for i, name in enumerate(RANK)}


def log(message):
    print(f"{time.strftime('%H:%M:%S')} {message}", flush=True)


def collapse_to_codes(labels):
    out = np.full(len(labels), RANK_CODE["Other"], dtype=np.int64)
    arr = np.asarray(labels, dtype=object)
    out[arr == "Inactivity"] = RANK_CODE["Inactivity"]
    out[arr == "Stable"] = RANK_CODE["Stable"]
    out[arr == FUEL_LABEL] = RANK_CODE["Fuel"]
    return out


def combine(p17, p4):
    base = p17[:, OTHER]
    cond = base / np.maximum(base.sum(axis=1, keepdims=True), 1e-12)
    out = np.zeros_like(p17)
    out[:, IDX["Inactivity"]] = p4[:, 0]
    out[:, IDX["Stable"]] = p4[:, 1]
    out[:, IDX[FUEL_LABEL]] = p4[:, 2]
    out[:, OTHER] = p4[:, 3:4] * cond
    return out / out.sum(axis=1, keepdims=True)


def load_proba(cutoff):
    return np.load(HERE / "probs" / f"{cutoff}.npz", allow_pickle=False)["proba"].astype(float)


def load_rank4(cutoff):
    return np.load(HERE / "rank4" / f"{cutoff}.npz", allow_pickle=False)["proba"].astype(float)


def main():
    started = time.time()
    with (HERE / "panel_plain.pkl").open("rb") as fh:
        panel = pickle.load(fh)
    vR = np.array(json.loads((HERE / "pool_r4full_vector.json").read_text())["vR"], dtype=float)

    import torch
    from tabpfn import TabPFNClassifier

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"tabpfn device {device}, cutoffs {len(panel['cutoffs'])}, features {len(panel['feature_cols'])}")

    def training_rows(latest):
        xs, ys = [], []
        for origin in panel["monthly_cutoffs"]:
            if origin > latest:
                continue
            block = panel["cutoffs"][origin]
            if block["y"] is None:
                continue
            xs.append(block["X"])
            ys.append(collapse_to_codes(block["y"]))
        x = np.concatenate(xs)
        y = np.concatenate(ys)
        if len(x) > CAP:
            x, y = x[-CAP:], y[-CAP:]
        return x, y

    (HERE / "tabpfn").mkdir(exist_ok=True)

    def member(cutoff):
        path = HERE / "tabpfn" / f"{cutoff}_tabpfn.npz"
        block = panel["cutoffs"][cutoff]
        if path.exists():
            payload = np.load(path, allow_pickle=False)
            if np.array_equal(payload["ids"].astype(str), block["ids"].astype(str)):
                return payload["proba"].astype(float)
        latest = str(np.datetime64(cutoff, "M") - 3) + "-01"
        x, y = training_rows(latest)
        log(f"  tabpfn {cutoff}: train rows {len(x)} (cap {CAP}), predict {len(block['X'])}")
        try:
            clf = TabPFNClassifier(device=device)
        except TypeError:
            clf = TabPFNClassifier()
        clf.fit(x, y)
        raw = clf.predict_proba(block["X"])
        out = np.zeros((len(block["X"]), 4))
        for j, c in enumerate(clf.classes_):
            out[:, int(c)] = raw[:, j]
        np.savez(path, proba=out, ids=block["ids"].astype(str))
        return out

    baseline, decision = {}, {"preregistration": "outputs/sprint17/proposal_tabpfn.json", "device": device}
    scores, gains = {}, {}
    first_seconds = None
    for cutoff in POOL:
        block = panel["cutoffs"][cutoff]
        y = np.asarray(block["y"], dtype=object)
        p17 = load_proba(cutoff)
        p4_base = load_rank4(cutoff)
        prod = combine(p17, p4_base)
        baseline[cutoff] = float(f1_score(y, np.asarray(CLASSES)[np.argmax(prod * vR, axis=1)], average="weighted", zero_division=0))
        if abs(baseline[cutoff] - BASELINE_GUARD[cutoff]) > 1e-9:
            raise RuntimeError(f"baseline mismatch at {cutoff}: {baseline[cutoff]} vs {BASELINE_GUARD[cutoff]}")
        t0 = time.time()
        p4 = 0.5 * p4_base + 0.5 * member(cutoff)
        if first_seconds is None:
            first_seconds = time.time() - t0
        prod = combine(p17, p4)
        scores[cutoff] = float(f1_score(y, np.asarray(CLASSES)[np.argmax(prod * vR, axis=1)], average="weighted", zero_division=0))
        gains[cutoff] = scores[cutoff] - baseline[cutoff]
        log(f"  tabpfn-mix {cutoff}: {scores[cutoff]:.6f}  gain {gains[cutoff]:+.6f}")
    decision["screen"] = {
        "baseline": baseline, "scores": scores, "gains": gains,
        "mean_gain": float(np.mean(list(gains.values()))),
        "min_gain": float(np.min(list(gains.values()))),
        "first_seconds": round(first_seconds, 1),
    }
    passed = (decision["screen"]["mean_gain"] >= SCREEN_GATE["mean_gain_min"]
              and decision["screen"]["min_gain"] >= SCREEN_GATE["per_origin_min"]
              and first_seconds <= 20 * 60)
    decision["screen"]["passed"] = bool(passed)
    log(f"SCREEN mean {decision['screen']['mean_gain']:+.6f} min {decision['screen']['min_gain']:+.6f} pass={passed}")

    if passed:
        fold_scores, fold_gains = {}, {}
        for cutoff in EVAL:
            block = panel["cutoffs"][cutoff]
            y = np.asarray(block["y"], dtype=object)
            p17, p4_base = load_proba(cutoff), load_rank4(cutoff)
            baseline[cutoff] = float(f1_score(y, np.asarray(CLASSES)[np.argmax(combine(p17, p4_base) * vR, axis=1)], average="weighted", zero_division=0))
            if abs(baseline[cutoff] - BASELINE_GUARD[cutoff]) > 1e-9:
                raise RuntimeError(f"baseline mismatch at {cutoff}")
            p4 = 0.5 * p4_base + 0.5 * member(cutoff)
            fold_scores[cutoff] = float(f1_score(y, np.asarray(CLASSES)[np.argmax(combine(p17, p4) * vR, axis=1)], average="weighted", zero_division=0))
            fold_gains[cutoff] = fold_scores[cutoff] - baseline[cutoff]
            log(f"  tabpfn-mix EVAL {cutoff}: {fold_scores[cutoff]:.6f}  gain {fold_gains[cutoff]:+.6f}")
        folds_ok = (all(g >= EVAL_GATE["per_origin_min"] for g in fold_gains.values())
                    and float(np.mean(list(fold_gains.values()))) >= EVAL_GATE["mean_gain_min"])
        decision["folds"] = {"scores": fold_scores, "gains": fold_gains, "passed": bool(folds_ok)}
        log(f"FOLDS pass={folds_ok}")
        if folds_ok:
            block = panel["cutoffs"][TEST]
            p4 = 0.5 * load_rank4(TEST) + 0.5 * member(TEST)
            prod = combine(load_proba(TEST), p4)
            pred = np.asarray(CLASSES)[np.argmax(prod * vR, axis=1)]
            with (HERE / "sasol_sprint16_r4full_vR.csv").open(newline="") as fh:
                source = list(csv.reader(fh))
            header, rows = source[0], source[1:]
            if header != ["ID", "CLV_fuel", "CLV_nonfuel", "Opportunity"]:
                raise RuntimeError(f"unexpected source header: {header}")
            if [r[0] for r in rows] != [str(i) for i in block["ids"]]:
                raise RuntimeError("December order mismatch with the source submission")
            changed = 0
            out_rows = []
            for r, p in zip(rows, pred):
                if r[3] != p:
                    changed += 1
                out_rows.append([r[0], r[1], r[2], p])
            path = HERE / "sasol_sprint17_tabpfnmix.csv"
            with path.open("w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(header)
                writer.writerows(out_rows)
            decision["exported"] = {"file": path.name, "changes_vs_best": changed}
            log(f"EXPORTED {path.name} with {changed} label changes")
    else:
        if first_seconds > 20 * 60:
            log("ABORT: first member fit exceeded 20 minutes")

    decision["seconds"] = round(time.time() - started, 1)
    (HERE / "tabpfn_decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    print("\n===== COPY THIS BACK =====")
    print(json.dumps(decision, indent=2))
    print("send back: tabpfn_decision.json, tabpfn/*.npz" + (", sasol_sprint17_tabpfnmix.csv" if decision.get("exported") else ""))


if __name__ == "__main__":
    main()
