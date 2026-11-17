"""Summarize output/multivar/<ds>/scale_expert runs -> logs/scale_expert_results.tsv.
Per run: overall PSNR, MB, primitives per expert, median max-axis scale per expert (voxels), routing."""
import glob, json, os
import numpy as np
from plyfile import PlyData

rows = []
for d in sorted(glob.glob("output/multivar/*/scale_expert*/*_seed*")):
    m = json.load(open(f"{d}/metrics.json"))
    r = json.load(open(f"{d}/routing.json"))
    X = json.load(open(f"{d}/volume.json"))["volume_shape"][0]
    med = []
    for k in range(3):
        f = f"{d}/expert_{k}/point_cloud.ply"
        if not os.path.exists(f):
            med.append("-"); continue
        p = PlyData.read(f)["vertex"]
        s = np.exp(np.stack([p["scale_0"], p["scale_1"], p["scale_2"]], 1)).max(1) / (2 / X)
        med.append(f"{np.median(s):.2f}")
    ds, exp = d.split("/")[2], d.split("/")[3].replace("scale_expert", "") + os.path.basename(d)
    route = " ".join(f"{v}:{''.join(str(k) for k, a in enumerate(r['active']) if a[i])}" for i, v in enumerate(r["variables"]))
    psnr = " ".join(f"{v}={m['psnr'][v]:.1f}" for v in r["variables"])
    rows.append([ds, exp, f"{m['overall_psnr']:.2f}", f"{m.get('overall_psnr_mask', float('nan')):.2f}",
                 f"{m['size_mb']:.2f}", ",".join(map(str, m["num_primitives"])), ",".join(med), route, psnr])
with open("logs/scale_expert_results.tsv", "w") as f:
    f.write("dataset\trun\toverall\toverall_mask\tMB\tN_per_expert\tmedian_scale_vox\trouting\tpsnr\n")
    for row in rows:
        f.write("\t".join(row) + "\n")
for row in rows:
    print("\t".join(row))
