"""See whether petrol versus diesel is new information for lapsed fuel CLV."""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
panel = pickle.loads((ROOT / "outputs" / "oct10" / "category_dynamics_panel.pkl").read_bytes())
raw = pd.read_csv(
    ROOT / "data" / "raw" / "train.csv",
    usecols=["DateTimeZA", "ID", "LineItemType", "ItemCategoryLevel2", "ItemCategoryLevel3", "ItemName", "Quantity"],
    dtype={"ID": str, "LineItemType": str, "ItemCategoryLevel2": str, "ItemCategoryLevel3": str, "ItemName": str},
)
fuel = raw.loc[raw.LineItemType.eq("Sale_Fuel") & raw.ItemCategoryLevel2.eq("Fuel")].copy()
print("level3")
print(fuel.ItemCategoryLevel3.value_counts().to_string())
print("names")
print(fuel.ItemName.value_counts().head(20).to_string())
fuel["time"] = pd.to_datetime(fuel.DateTimeZA, format="mixed")
fuel["qty"] = pd.to_numeric(fuel.Quantity, errors="coerce").clip(lower=0)
fuel = fuel.loc[fuel.time.lt("2025-09-01")]
fuel["diesel"] = fuel.ItemCategoryLevel3.str.contains("Diesel", case=False, na=False) | fuel.ItemName.str.contains("Diesel", case=False, na=False)
diesel_qty = fuel.loc[fuel.diesel].groupby("ID").qty.sum()
total_qty = fuel.groupby("ID").qty.sum()
share = (diesel_qty / total_qty).reindex(total_qty.index).fillna(0)
frame = panel["frames"]["2025-09-01"]
y = panel["labels"]["2025-09-01"].reindex(frame.index).CLV_fuel.to_numpy()
x = share.reindex(frame.index).fillna(0).to_numpy()
rec = frame.fuel_recency.to_numpy()
m = (rec > 30) & (rec <= 90)
print("diesel share mean", float(x.mean()), "lapsed mean", float(x[m].mean()))
print("corr all", float(np.corrcoef(x, y)[0, 1]), "lapsed", float(np.corrcoef(x[m], y[m])[0, 1]))
# Residual correlation after fuel_per_tx and fuel_z_q1.
base = np.column_stack([np.ones(m.sum()), frame.fuel_per_tx.to_numpy()[m], frame.fuel_z_q1.to_numpy()[m]])
coef, *_ = np.linalg.lstsq(base, y[m], rcond=None)
resid = y[m] - base @ coef
coef2, *_ = np.linalg.lstsq(base, x[m], rcond=None)
resid_x = x[m] - base @ coef2
print("partial corr lapsed", float(np.corrcoef(resid_x, resid)[0, 1]))
