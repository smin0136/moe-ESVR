"""Per-expert size of the exact (uniform) and non-uniform budget runs.

Usage: python3 logs/summarize_nonuni_experts.py  -> prints and writes logs/nonuni_experts.tsv
An expert's size counts its primitives x (10 geometry floats + its active channels).
"""
import json
import os

ROOT = "output/multivar"
GRID = [("ionization_ds2", ["separate", "K2_lam3e-1", "K4sep_lam1e-1"], [2, 4, 8]),
        ("combustion_ds2", ["separate", "K2_lam3e-1", "K4sep_lam3e-1"], [2, 4, 8]),
        ("rocom_masked", ["separate", "K4_lam1e-2", "K8sep_lam1e-2"], [4, 8, 16])]
MODES = ["exact", "pool", "usage", "ratio"]


def experts(run_dir):
    m = json.load(open(os.path.join(run_dir, "metrics.json")))
    channels = {}
    for name, pairs in m["routing"].items():
        pairs = [pairs] if isinstance(pairs[0], int) else pairs
        for k, _ in pairs:
            channels.setdefault(k, []).append(name)
    out = []
    for k, n in enumerate(m["num_primitives"]):
        if k in channels and n > 0:
            out.append((k, n, n * (10 + len(channels[k])) * 4 / 1024**2, channels[k]))
    return m, out


rows = []
for ds, methods, targets in GRID:
    for method in methods:
        for t in targets:
            for mode in MODES:
                run = (f"{ROOT}/{ds}/budget_exact/{method}_mb{t}" if mode == "exact"
                       else f"{ROOT}/{ds}/budget_nonuni/{method}_mb{t}_{mode}")
                m, ex = experts(run)
                sizes = " ".join(f"E{k}:{mb:.2f}" for k, _, mb, _ in ex)
                rows.append([ds, method, str(t), mode, f"{m['overall_psnr']:.2f}", f"{m['size_mb']:.2f}",
                             str(len(ex)), sizes, " ".join(f"E{k}:{n}" for k, n, _, _ in ex),
                             " ".join(f"E{k}:{','.join(v)}" for k, _, _, v in ex), run])
with open("logs/nonuni_experts.tsv", "w") as f:
    f.write("\t".join(["dataset", "method", "target_mb", "mode", "overall", "size_mb", "num_experts",
                       "expert_mb", "expert_primitives", "expert_variables", "run_dir"]) + "\n")
    for r in rows:
        f.write("\t".join(r) + "\n")
for r in rows:
    print("\t".join(r[:8]))
