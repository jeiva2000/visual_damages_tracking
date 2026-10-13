import json, math, collections
ET="/mnt/Disk2/RT-DETR_4/RT-DETR/rtdetrv2_pytorch/eval_tracking"
SEEN={"touaregmedina","classiccar","video6","chevroletazul","video2","video4","4x4verde","chevroletdorado","video3","video5"}
def wilson(k,n,z=1.96):
    if n==0: return (0,0)
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return 100*(c-h),100*(c+h)
out={}
for tr in ["bytetrack","botsort","ocsort","deepocsort"]:
    g=json.load(open(f"{ET}/reid_gap_report_{tr}_alldata/reid_gap_report.json"))
    out[tr]={}
    for tag,keep in [("all",lambda s:True),("excl",lambda s:s not in SEEN)]:
        gg=[x for x in g if keep(x["sequence"])]
        c=collections.Counter(x["outcome"] for x in gg); ev=len(gg)-c["unevaluated"]
        r=c["reidentified"]; lo,hi=wilson(r,ev)
        out[tr][tag]=dict(total=len(gg),evaluated=ev,reid=r,new=c["new_id_born"],swap=c["id_swap"])
        print(f"{tr:10s} {tag:4s} total={len(gg)} eval={ev} ({100*ev/len(gg):.1f}%) reid={r} ({100*r/ev:.1f}%) CI=[{lo:.1f},{hi:.1f}] new={c['new_id_born']} ({100*c['new_id_born']/ev:.1f}%) swap={c['id_swap']} ({100*c['id_swap']/ev:.1f}%)")
json.dump(out,open(f"{ET}/exp_exclude_seen/tracker_base_reid_fail_all_vs_excl.json","w"),indent=1)
