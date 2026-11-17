"""Results of the top-k capacity, exact-budget and non-uniform-budget grids
(logs/make_topk_cap_jobs.sh, make_budget_exact_jobs.sh, make_budget_nonuni_jobs.sh).

Usage: python3 logs/summarize_cap_budget.py  -> prints and writes logs/cap_budget_results.tsv
"""
import glob
import json
import os

ROOT = "output/multivar"
DATASETS = ["ionization_ds2", "combustion_ds2", "rocom_masked", "snu_down_down"]
GRIDS = ["topk_cap", "budget_exact", "budget_nonuni"]


def usage(m):
    routing = m.get("routing") or {}
    out = []
    for name in m["psnr"]:
        pairs = routing.get(name, [])
        pairs = [pairs] if pairs and isinstance(pairs[0], int) else pairs
        out.append(f"{name}:{''.join(str(k) for k, _ in pairs)}")
    return " ".join(out)


rows = []
for ds in DATASETS:
    for grid in GRIDS:
        for path in sorted(glob.glob(os.path.join(ROOT, ds, grid, "*", "metrics.json"))):
            m = json.load(open(path))
            mask = m.get("overall_psnr_mask")
            rows.append([ds, grid, os.path.basename(os.path.dirname(path)), f"{m['overall_psnr']:.2f}",
                         f"{mask:.2f}" if mask is not None else "-", f"{m['size_mb']:.2f}",
                         ",".join(str(n) for n in m["num_primitives"]), usage(m)]
                        + [f"{n}={p:.1f}" for n, p in m["psnr"].items()])
header = ["dataset", "grid", "run", "overall", "mask", "size_mb", "per_expert", "usage", "psnr_per_var"]
with open("logs/cap_budget_results.tsv", "w") as f:
    f.write("\t".join(header) + "\n")
    for r in rows:
        f.write("\t".join(r[:8] + [" ".join(r[8:])]) + "\n")
for r in rows:
    print("\t".join(r))
