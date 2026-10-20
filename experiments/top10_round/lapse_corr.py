import pickle
from pathlib import Path
import numpy as np
p = pickle.loads((Path(__file__).resolve().parents[2] / "outputs" / "oct10" / "category_dynamics_panel.pkl").read_bytes())
frame = p["frames"]["2025-09-01"]
y = p["labels"]["2025-09-01"].reindex(frame.index).CLV_fuel.to_numpy()
m = (frame.fuel_recency.to_numpy() > 30) & (frame.fuel_recency.to_numpy() <= 90)
print("n", int(m.sum()))
names = [c for c in frame.columns if "fuel" in c or "recency" in c or "overdue" in c or "gap" in c or "cadence" in c or "bgnbd" in c]
scored = []
for name in names:
    x = frame[name].to_numpy()[m]
    if np.nanstd(x) == 0:
        continue
    scored.append((abs(float(np.corrcoef(x, y[m])[0, 1])), float(np.corrcoef(x, y[m])[0, 1]), name))
for row in sorted(scored, reverse=True)[:25]:
    print(f"{row[1]:+.4f} {row[2]}")
