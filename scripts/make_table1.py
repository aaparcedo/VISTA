"""Build Table 1 (and the fine-grained table) from per-sample results.

    python scripts/make_table1.py --results results/ --annotations data/vista_annotations.jsonl --out tables/

Results are the JSON Lines written by vista.run_eval (any number of files, any sharding). Rows are joined to the
annotations by sample_id, so each (sample, query) is counted exactly once per model.

Columns (as in the paper):
  R, F, R&F : mean mvIoU over referral rows, freeform rows, and all rows (pooled)
  S, T      : mean over freeform rows labelled Spatial / Temporal
  AA..OO    : mean over freeform rows of each involved-entity class
Taxonomy labels are assigned to freeform queries only, so S/T and entity columns use freeform rows.
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import pandas as pd

ENTITY = {"AA": "Animal-Animal", "AO": "Animal-Object", "HA": "Human-Animal", "HH": "Human-Human",
          "HO": "Human-Object", "HS": "Human-Self", "NI": "No Interaction", "OO": "Object-Object"}
EXPECTED_PAIRS = 11814


def load_results(results_dir):
    rows, seen, dups = [], set(), 0
    for fp in sorted(glob.glob(os.path.join(results_dir, "**", "*.jsonl"), recursive=True)):
        with open(fp) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                key = (r["model"], r["sample_id"], r["task_type"])
                if key in seen:
                    dups += 1
                    continue
                seen.add(key)
                rows.append({k: r[k] for k in ("model", "sample_id", "task_type", "dataset", "mvIoU")})
    if dups:
        print(f"note: ignored {dups} duplicate (model, sample, task) rows")
    return pd.DataFrame(rows)


def main(args):
    ann = pd.DataFrame([json.loads(l) for l in open(args.annotations)]).set_index("sample_id")
    res = load_results(args.results)
    res = res.join(ann[["label_st", "label_entity", "labels_fine"]], on="sample_id")

    table, fine, coverage = {}, {}, {}
    for model, g in res.groupby("model"):
        ff = g[g.task_type == "freeform"]
        row = {"R": g[g.task_type == "referral"].mvIoU.mean() * 100, "F": ff.mvIoU.mean() * 100, "R&F": g.mvIoU.mean() * 100,
               "S": ff[ff.label_st == "Spatial"].mvIoU.mean() * 100, "T": ff[ff.label_st == "Temporal"].mvIoU.mean() * 100}
        for col, cls in ENTITY.items():
            row[col] = ff[ff.label_entity == cls].mvIoU.mean() * 100
        table[model] = row
        per_cat = defaultdict(list)
        for _, r in ff.iterrows():
            for c in r.labels_fine or []:
                per_cat[c].append(r.mvIoU)
        fine[model] = {c: 100 * sum(v) / len(v) for c, v in per_cat.items()}
        coverage[model] = {"pairs": len(g), "of": EXPECTED_PAIRS, "complete": len(g) == EXPECTED_PAIRS,
                           **{f"{d}/{t}": n for (d, t), n in g.groupby(["dataset", "task_type"]).size().items()}}

    os.makedirs(args.out, exist_ok=True)
    t = pd.DataFrame(table).T
    t[["R", "F", "R&F"]] = t[["R", "F", "R&F"]].round(2)
    t[[c for c in t.columns if c not in ("R", "F", "R&F")]] = t[[c for c in t.columns if c not in ("R", "F", "R&F")]].round(1)
    t.to_csv(os.path.join(args.out, "table1.csv"))
    pd.DataFrame(fine).T.round(1).sort_index(axis=1).to_csv(os.path.join(args.out, "table_finegrained.csv"))
    json.dump(coverage, open(os.path.join(args.out, "coverage.json"), "w"), indent=1)
    print(t.to_string())
    incomplete = [m for m, c in coverage.items() if not c["complete"]]
    if incomplete:
        print(f"\nWARNING: incomplete coverage (fewer than {EXPECTED_PAIRS} pairs) for: {incomplete}; see coverage.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--annotations", default="data/vista_annotations.jsonl")
    p.add_argument("--out", default="tables")
    main(p.parse_args())
