import pickle
from pathlib import Path
p = pickle.loads((Path(__file__).resolve().parents[2] / "outputs" / "oct10" / "category_dynamics_panel.pkl").read_bytes())
cols = p["feature_cols"]
print("n", len(cols))
for c in cols:
    if c.startswith("m") or "ses" in c or "dow" in c or "hour" in c or "location" in c or "weekend" in c:
        print(c)
