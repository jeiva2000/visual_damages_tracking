import json,collections,os,sys
ET="/mnt/Disk2/RT-DETR_4/RT-DETR/rtdetrv2_pytorch/eval_tracking"; H=f"{ET}/exp_exclude_seen/hard_cases"
TH=0.9405
CASES=[("video_75wAEktItLIYWrWOW7E_11",2,183),("video_0vbcxRjnfa70hc8AVvt_11",1,170),("video_0XWkyECiQbjitGSpVQK_11",4,292),("video_Ve7DX8i8A6RX5h8OyFt_11",5,31),("video_0jGlrs0wJs3utJftEoD_11",2,289),("video_0jGlrs0wJs3utJftEoD_11",2,347),("video_1Vl7JUY8fBQhAph15I7_11",1,19),("video_rICjFXthyXrdrgr58oE_11",1,169),("video_kYUj2rFivaSQMwnjjGp_11",2,234),("video_Mpn2iv5EwC8BQM4Guf2_11",1,78),("video_Mpn2iv5EwC8BQM4Guf2_11",2,224)]
def mot(p):
    d=collections.defaultdict(dict)
    for l in open(p):
        a=l.split(",")
        if len(a)>=6: d[int(float(a[0]))][int(float(a[1]))]=tuple(map(float,a[2:6]))
    return d
def iou(a,b):
    iw=max(0,min(a[0]+a[2],b[0]+b[2])-max(a[0],b[0]));ih=max(0,min(a[1]+a[3],b[1]+b[3])-max(a[1],b[1]));i=iw*ih
    u=a[2]*a[3]+b[2]*b[3]-i;return i/u if u>0 else 0
out=[]
for t in("botsort","bytetrack"):
    G=json.load(open(f"{H}/c4_{t}/gap_outcomes.json"))
    for s,gid,fb in CASES:
        g=[x for x in G if x["sequence"]==s and x["gt_identity_id"].endswith(f"__{gid}") and x["frame_before"]==fb][0]
        if g["outcome"]=="unevaluated": continue
        fa=g["frame_after"];bid=g["pred_id_before"]
        gt=mot(f"{ET}/all_dataset_view/all/{s}/gt/gt.txt");pr=mot(f"{ET}/exp_live_excl/c4_live_{t}_alldata/txt/{s}.txt")
        rows=[json.loads(l) for l in open(f"{H}/sims/{t}_{s}.jsonl")]
        att=[]
        for r in rows:
            f=r["frame"]
            if f<fb or f>fa+60 or gid not in gt.get(f,{}): continue
            for i,rid in enumerate(r["raw_ids"]):
                m=r["res"][i];fin=m if m is not None else rid
                b=pr.get(f,{}).get(fin)
                if b and iou(b,gt[f][gid])>=.5:
                    sm=dict(zip(r["lost"],r["sims"][i]))
                    att.append((f,rid,m,sm))
        print(f"\n{t} {s[:14]} gt{gid} {fb}->{fa} outcome={g['outcome']} gap={g['gap_len']} target_bank_id={bid}")
        for f,rid,m,sm in att[:3]:
            best=max(sm,key=sm.get);tg=sm.get(bid)
            rank=None if tg is None else 1+sum(v>tg for v in sm.values())
            nabove=sum(v>=TH for v in sm.values())
            print(f"  f{f} newborn_raw={rid} -> {'MATCH '+str(m) if m is not None else 'nomatch'} | target sim={tg} rank={rank}/{len(sm)} | best id{best} sim={sm[best]:.4f} | n_bank>=thr={nahove if False else nabove}")
        if not att: print("  (sin intento de reid solapado con el GT objetivo)")
