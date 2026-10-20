"""Does the weekly block correlate with the shipped fuel residual on the pool?

No fit. June and September are not read. A near-zero correlation means a
weekly fuel refit is not the next experiment.
"""
import pickle

import numpy as np

from fit_weekly_rank4 import OCT, WEEKLY

POOL = ["2025-04-01", "2025-05-01", "2025-07-01", "2025-08-01"]


def main():
    panel = pickle.loads((OCT / "category_dynamics_panel.pkl").read_bytes())
    for cutoff in POOL:
        frame = panel["frames"][cutoff]
        y = panel["labels"][cutoff].reindex(frame.index).CLV_fuel.to_numpy(dtype=float)
        direct = np.load(OCT / f"{cutoff}_CLV_fuel_category_values.npy")
        hurdle = np.load(OCT / f"{cutoff}_CLV_fuel_two_part.npy")
        residual = np.load(OCT / f"{cutoff}_fuel_residual.npy")
        if not (len(y) == len(direct) == len(hurdle) == len(residual)):
            raise RuntimeError(f"{cutoff} fuel parts do not match the frame")
        pred = 0.40 * direct + 0.35 * hurdle + 0.25 * residual
        err = y - pred
        payload = np.load(WEEKLY / f"{cutoff}.npz", allow_pickle=False)
        names = payload["names"].astype(str).tolist()
        x = payload["X"]
        print(cutoff, "rmse", round(float(np.sqrt(np.mean(err ** 2))), 4), "n", len(y))
        for name in ("wk_fuel_r_shape", "wk_fuel_r_last4", "wk_fuel_runrate_gap", "wk_fuel_weeks_last8", "wk00_fuel_l"):
            column = x[:, names.index(name)]
            corr = float(np.corrcoef(column, err)[0, 1])
            print(f"  {name:24s} corr {corr:+.4f}")


if __name__ == "__main__":
    main()
