"""Evaluate the registered blend under each decode vector on June/September.

Evaluation only: vectors were tuned on non-evaluation origins; these folds
never entered any objective. Writes outputs/sprint16/blend_vector_eval.csv.
"""
import json

import numpy as np
import pandas as pd

from evaluate_saved import CLASSES, MULT, OUT, rank4_combine, load_labels, load_proba, load_rank4, weighted_f1


def main():
    vectors = {"accepted": MULT}
    payload = json.loads((OUT / "multiplier_transfer.json").read_text())
    for row in payload["rows"]:
        vectors[row["vector"]] = np.array(row["values"], dtype=float)

    rows = []
    for cutoff in ["2025-06-01", "2025-09-01"]:
        ids, y = load_labels(cutoff)
        n = len(ids)
        p_cat = load_proba("category_dynamics", cutoff, n)
        p_early = load_proba("early_history", cutoff, n)
        r4 = load_rank4(cutoff, n)
        r4full = rank4_combine(p_cat, r4, "full")
        blend = 0.4 * p_cat + 0.4 * p_early + 0.2 * r4full
        for name, vec in vectors.items():
            f_cat = weighted_f1(np.argmax(p_cat * vec, axis=1), y)
            f_blend = weighted_f1(np.argmax(blend * vec, axis=1), y)
            rows.append({"origin": cutoff, "vector": name, "catdyn_f1": f_cat, "blend_f1": f_blend})
    table = pd.DataFrame(rows).pivot(index="vector", columns="origin", values="blend_f1")
    cat = pd.DataFrame(rows).pivot(index="vector", columns="origin", values="catdyn_f1")
    table.to_csv(OUT / "blend_vector_eval.csv")
    print("blend F1 by decode vector:")
    print(table.round(5).to_string())
    print("\ncatdyn F1 by decode vector:")
    print(cat.round(5).to_string())


if __name__ == "__main__":
    main()
