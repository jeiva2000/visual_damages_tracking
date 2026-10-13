"""
EXPERIMENTO AISLADO -- tab:kfold_identity_acc y McNemar excluyendo los eventos
de las 10 secuencias que probablemente vio el detector actual.

No re-ejecuta nada: usa los resultados POR EVENTO ya guardados (mismos que
respaldan la tabla y el McNemar del paper) y el fold_assignment de la k-fold.
  - precisión por fold = aciertos / eventos evaluables del fold (sin None), y
    luego media +- desviación entre folds -- mismo estadístico que la tabla.
  - McNemar exacto (pares discordantes, todos los folds juntos).
Control: con TODAS las secuencias debe reproducir los valores de la tabla.
Umbrales: los ya calibrados por fold (calibración en train/val de cada fold,
que sí incluía esas secuencias) -- ver nota en la salida.
"""
import json
import os
from math import comb

import numpy as np

VDT = "/mnt/Disk2/RT-DETR_4/RT-DETR/rtdetrv2_pytorch/visual_damages_tracking"
SEEN = {"touaregmedina", "classiccar", "video6", "chevroletazul", "video2",
        "video4", "4x4verde", "chevroletdorado", "video3", "video5"}
TRACKERS = ["bytetrack", "botsort", "ocsort", "deepocsort"]


def jload(name):
    return json.load(open(os.path.join(VDT, name)))


def pick(d, *prefixes):
    for p in prefixes:
        for k in d:
            if k == p or k.startswith(p):
                return d[k]
    raise KeyError(prefixes)


def per_case(tr):
    """{metodo: {case_key: bool|None}} para un tracker, desde los json por evento."""
    cl = jload(f"mcnemar_c4_vs_clipvehicleid_{tr}.json")
    os_ = jload(f"mcnemar_c4_vs_osnet_postfix_{tr}.json")
    pf = jload(f"mcnemar_c4_vs_p4fpn_{tr}.json")
    hd = jload(f"mcnemar_c4_vs_heads_{tr}.json")
    oc = tr in ("ocsort", "deepocsort")
    return {
        "DINOv2": pick(cl, "dinov2"),
        "OSNet": pick(os_, "osnet"),
        "CLIP-VehicleID": pick(cl, "clipvehicleid"),
        "C4": cl["c4_roi_oc" if oc else "p4_roi"],
        "C4+H1": pick(hd, "c4_v6head_oc" if oc else "p4_v6head"),
        "C4+H2": pick(hd, "c4_intermedio_head_oc" if oc else "p4_intermedio_head"),
        "C4+H3": pick(hd, "c4_v7head_oc" if oc else "p4_v7head"),
        "C4+H4": pick(hd, "c4_puretriplet_head_oc" if oc else "p4_puretriplet_head"),
        "P4-FPN": pick(pf, "p4fpn"),
    }


def fold_of_seq(tr):
    d = "c4_roi_kfold_baseline_ocsort_deepocsort" if tr in ("ocsort", "deepocsort") else "p4_roi_kfold_baseline"
    return {s: fr["fold"] for fr in jload(f"{d}/kfold_results.json") for s in fr["test_seqs"]}


def fold_stats(cases, fold_of, keep):
    per_fold = {}
    for k, v in cases.items():
        seq = k.split("__")[0]
        if v is None or not keep(seq):
            continue
        per_fold.setdefault(fold_of[seq], []).append(bool(v))
    accs = [np.mean(v) for _, v in sorted(per_fold.items())]
    n = sum(len(v) for v in per_fold.values())
    return float(np.mean(accs)), float(np.std(accs)), n


def mcnemar(a, b, keep):
    ks = [k for k in set(a) & set(b) if a[k] is not None and b[k] is not None and keep(k.split("__")[0])]
    n10 = sum(1 for k in ks if a[k] and not b[k]); n01 = sum(1 for k in ks if b[k] and not a[k]); nd = n10 + n01
    p = min(1.0, 2 * sum(comb(nd, i) for i in range(min(n10, n01) + 1)) / 2 ** nd) if nd else 1.0
    return n10, n01, p


def main():
    out = {"nota": "umbrales calibrados por fold en la k-fold original (que incluía las 10 secuencias en train/val)"}
    keep_all = lambda s: True
    keep_ex = lambda s: s not in SEEN
    for tr in TRACKERS:
        pc, fo = per_case(tr), fold_of_seq(tr)
        out[tr] = {}
        for m, cases in pc.items():
            ma, sa, na = fold_stats(cases, fo, keep_all)
            me, se, ne = fold_stats(cases, fo, keep_ex)
            out[tr][m] = {"all72": [ma, sa, na], "excl": [me, se, ne]}
        out[tr]["mcnemar_excl"] = {o: mcnemar(pc["C4"], pc[o], keep_ex) for o in pc if o != "C4"}
        out[tr]["mcnemar_all"] = {o: mcnemar(pc["C4"], pc[o], keep_all) for o in pc if o != "C4"}
    os.makedirs(os.path.join(VDT, "exp_exclude_seen"), exist_ok=True)
    path = os.path.join(VDT, "exp_exclude_seen", "kfold_all72_vs_excl.json")
    json.dump(out, open(path, "w"), indent=1)

    fmt = lambda x: f"{x[0]:.4f}+-{x[1]:.4f}"
    print("precisión k-fold (media +- desv entre folds) | all72 -> excl")
    for m in out["bytetrack"]:
        if m.startswith("mcnemar"):
            continue
        print(f"  {m:15s} " + " | ".join(f"{tr[:4]} {fmt(out[tr][m]['all72'])} -> {fmt(out[tr][m]['excl'])}" for tr in TRACKERS))
    print("eventos: " + " ".join(f"{tr} {out[tr]['C4']['all72'][2]}->{out[tr]['C4']['excl'][2]}" for tr in TRACKERS))
    print("\nMcNemar C4 vs X, SIN las 10 secuencias: p (n10/n01)")
    f2 = lambda r: f"{'<0.001' if r[2] < 0.001 else f'{r[2]:.3f}'} ({r[0]}/{r[1]})"
    for o in out["bytetrack"]["mcnemar_excl"]:
        print(f"  vs {o:15s} " + " | ".join(f2(out[tr]["mcnemar_excl"][o]) for tr in TRACKERS))
    print(f"\n[OK] {path}")


if __name__ == "__main__":
    main()
