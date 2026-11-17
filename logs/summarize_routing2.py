"""Results of the constrained channel-routing grid (logs/make_routing2_jobs.sh).

Usage: python3 logs/summarize_routing2.py [dataset ...] [--tsv out.tsv]
Prints one table per dataset with earlier reference runs, then writes every
row to a TSV for plotting.
"""
import glob
import json
import os
import re
import sys

ROOT = "output/multivar"
REFERENCES = {
    "ionization_ds2": ["shared", "channel_shared_lam1e-1", "channel_shared_lam3e-1",
                       "channel_K2_lam1e-1", "channel_K2_lam3e-1", "channel_K2_lam1"],
    "combustion_ds2": ["shared", "channel_shared_lam1e-1", "channel_shared_lam3e-1",
                       "channel_K2_lam1e-1", "channel_K2_lam3e-1", "channel_K2_lam1"],
    "snu_down_down": ["t0_repro/shared"],
    "rocom_masked": ["maxgrad/shared", "channel/shared_lam1e-2", "channel/K4_off0.02_lam1e-2",
                     "channel/K4_off0.02_lam3e-2", "rd/K4_lam1e-2_p0.5"],
}
TAG = re.compile(r"(?P<init>shared|K\d+(?:sep)?)_lam(?P<lam>[^_]+)(?:_top(?P<k>\d)(?P<mode>exact|max)?)?(?:_mb(?P<mb>[\d.]+))?$")


def row(dataset, run_dir, label):
    path = os.path.join(run_dir, "metrics.json")
    if not os.path.exists(path):
        return None
    m = json.load(open(path))
    variables = list(m["psnr"])
    routing = m.get("routing") or {}
    experts_per_var = [len(routing[n]) for n in variables if isinstance(routing.get(n), list)]
    tag = TAG.match(label.split("/")[-1])
    parts = tag.groupdict() if tag else {}
    return {
        "dataset": dataset, "run": label,
        "init": parts.get("init") or "-", "lambda": parts.get("lam") or "-",
        "topk": (parts.get("k") or "-") + ((parts.get("mode") or "exact") if parts.get("k") else ""),
        "target_mb": parts.get("mb") or "-",
        "overall": m["overall_psnr"], "mean": m["mean_psnr"], "mask": m.get("overall_psnr_mask"),
        "size_mb": m["size_mb"], "N": sum(m["num_primitives"]),
        "per_expert": ",".join(str(n) for n in m["num_primitives"]),
        "experts_per_var": sum(experts_per_var) / len(experts_per_var) if experts_per_var else float("nan"),
        "usage": " ".join(f"{n}:{''.join(str(k) for k, _ in routing[n])}" for n in variables
                          if isinstance(routing.get(n), list)),
        "psnr": m["psnr"],
    }


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    tsv = sys.argv[sys.argv.index("--tsv") + 1] if "--tsv" in sys.argv else "logs/routing2_results.tsv"
    datasets = [a for a in args if a != tsv] or list(REFERENCES)
    rows = []
    for ds in datasets:
        ds_rows = [r for r in (row(ds, os.path.join(ROOT, ds, ref), "ref:" + ref) for ref in REFERENCES[ds]) if r]
        for d in sorted(glob.glob(os.path.join(ROOT, ds, "routing2", "*"))):
            r = row(ds, d, os.path.basename(d))
            if r:
                ds_rows.append(r)
        if not ds_rows:
            continue
        print(f"\n### {ds}  ({len(ds_rows)} runs)")
        print(f"{'run':42s} {'overall':>7s} {'mask':>6s} {'MB':>6s} {'N':>8s} {'exp/var':>7s}  usage (variable: experts)")
        for r in ds_rows:
            mask = f"{r['mask']:.2f}" if r["mask"] else ""
            print(f"{r['run']:42s} {r['overall']:7.2f} {mask:>6s} {r['size_mb']:6.2f} {r['N']:8d} "
                  f"{r['experts_per_var']:7.2f}  {r['usage']}")
        rows += ds_rows
    with open(tsv, "w") as f:
        keys = ["dataset", "run", "init", "lambda", "topk", "target_mb", "overall", "mean", "mask",
                "size_mb", "N", "per_expert", "experts_per_var", "usage"]
        f.write("\t".join(keys) + "\n")
        for r in rows:
            f.write("\t".join(str(r[k]) for k in keys) + "\n")
    print(f"\nwrote {tsv}")


if __name__ == "__main__":
    main()
