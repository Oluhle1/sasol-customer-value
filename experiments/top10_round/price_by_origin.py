import pickle
from pathlib import Path
p = pickle.loads((Path(__file__).resolve().parents[2] / "outputs" / "oct10" / "category_dynamics_panel.pkl").read_bytes())
for cutoff, frame in sorted(p["frames"].items()):
    s = frame["price_index"]
    print(cutoff, "rows", len(frame), "price", float(s.iloc[0]), "unique", int(s.nunique()), "fuel_z_q1", float(frame.fuel_z_q1.mean()))
