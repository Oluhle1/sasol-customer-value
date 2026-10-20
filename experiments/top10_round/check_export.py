import pandas as pd
from pathlib import Path
root = Path(__file__).resolve().parents[2]
a = pd.read_csv(root / "submissions" / "sasol_oct10_small_fuel_values.csv", dtype={"ID": str})
b = pd.read_csv(root / "submissions" / "sasol_top10_rank4.csv", dtype={"ID": str})
print(len(b), list(b.columns))
print("ids", b.ID.equals(a.ID))
print("fuel_max_abs", float((b.CLV_fuel - a.CLV_fuel).abs().max()))
print("shop_max_abs", float((b.CLV_nonfuel - a.CLV_nonfuel).abs().max()))
print("opportunity_changes", int((b.Opportunity != a.Opportunity).sum()))
print("labels", int(b.Opportunity.nunique()))
