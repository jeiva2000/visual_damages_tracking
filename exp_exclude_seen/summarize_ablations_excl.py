import os, re
V="/mnt/Disk2/RT-DETR_4/RT-DETR/rtdetrv2_pytorch/visual_damages_tracking"; A=V+"/exp_exclude_seen/abl"
def res(path):
    if not os.path.exists(path): return None
    t=open(path,errors="ignore").read(); i=t.rfind("[RESUMEN K-FOLD]")
    if i<0: return None
    out={}
    for l in t[i:].splitlines()[1:12]:
        m=re.match(r"(\w+)\s+([\d.]+) \+/- ([\d.]+)\s+([\d.]+) \+/- ([\d.]+)",l.strip())
        if m: out[m.group(1)]=tuple(float(m.group(k)) for k in (2,3,4,5))
    return out
def f(r,src,col): 
    if r is None or src not in r: return "  n/a  "
    return "%.4f±%.4f"%(r[src][col],r[src][col+1])
E=lambda n: A+f"/{n}_EXCL/kfold.log"
print("== gap length (identity acc)  orig -> excl")
for b,lab in [("lt3","<3"),("3-8","3-8"),("9-32","9-32"),("gt32",">32")]:
    o1=res(V+f"/kfold_c4_gaplen_{b}_bytetrack_botsort.log"); o2=res(V+f"/kfold_c4_gaplen_{b}_ocsort_deepocsort.log")
    e1=res(E(f"gaplen_{b}_bt_bs")); e2=res(E(f"gaplen_{b}_oc_doc"))
    print(lab.ljust(5)," | ".join("%s %s -> %s"%(t[:4],f(o,t,2)[:6],f(e,t,2)[:6]) for t,o,e in [("bytetrack",o1,e1),("botsort",o1,e1),("ocsort",o2,e2),("deepocsort",o2,e2)]))
print("== threshold (bytetrack identity / synthetic match)")
for enc,ts in [("p4",["0.50","0.65","0.80","0.90","0.96","0.99"]),("p4fpn",["0.30","0.45","0.60","0.85","0.92","0.97"])]:
    for t in ts:
        o=res(V+f"/kfold_threshold_ablation_{enc}_t{t}.log"); e=res(E(f"thr_{'c4' if enc=='p4' else enc}_t{t}"))
        sk=[k for k in (o or {}) if k.startswith("synthetic")]; sk=sk[0] if sk else "synthetic_test"
        print(enc,t,"id",f(o,"bytetrack",2)[:6],"->",f(e,"bytetrack",2)[:6],"| match",f(o,sk,0)[:6],"->",f(e,sk,0)[:6])
print("== wide (bytetrack identity)")
for s in [1,2,4,8]:
    print(s, f(res(V+f"/kfold_c4_ablation_wide{s}_postfix.log"),"bytetrack",2),"->",f(res(E(f"wide{s}")),"bytetrack",2))
print("6 (C4 EXCL)", "->", f(res(V+"/exp_exclude_seen/p4_roi_kfold_baseline_EXCL/kfold.log"),"bytetrack",2))
print("== fusion (bytetrack identity)")
for x in ["near","wide","max","concat"]:
    print(x, f(res(V+f"/kfold_c4_ablation_fusion_{x}_reimpl.log"),"bytetrack",2),"->",f(res(E(f"fusion_{x}")),"bytetrack",2))
print("== inverted calib (identity)")
for enc in ["c4","p4fpn"]:
    o=res(V+f"/kfold_inverted_calib_{enc}_4t.log"); e=res(E(f"inverted_{enc}"))
    print(enc," | ".join("%s %s -> %s"%(t[:4],f(o,t,2),f(e,t,2)) for t in ["bytetrack","botsort","ocsort","deepocsort"]))
