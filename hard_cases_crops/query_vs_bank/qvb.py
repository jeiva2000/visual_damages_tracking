import json,os,re,cv2,numpy as np,collections
ET="/mnt/Disk2/RT-DETR_4/RT-DETR/rtdetrv2_pytorch/eval_tracking"; H=f"{ET}/exp_exclude_seen/hard_cases"
OUT=f"{H}/query_vs_bank2"; os.makedirs(OUT,exist_ok=True)
CASES=[("video_75wAEktItLIYWrWOW7E_11",2,183),("video_0vbcxRjnfa70hc8AVvt_11",1,170),("video_0XWkyECiQbjitGSpVQK_11",4,292),("video_Ve7DX8i8A6RX5h8OyFt_11",5,31),("video_0jGlrs0wJs3utJftEoD_11",2,289),("video_0jGlrs0wJs3utJftEoD_11",2,347),("video_1Vl7JUY8fBQhAph15I7_11",1,19),("video_rICjFXthyXrdrgr58oE_11",1,169),("video_kYUj2rFivaSQMwnjjGp_11",2,234),("video_Mpn2iv5EwC8BQM4Guf2_11",1,78),("video_Mpn2iv5EwC8BQM4Guf2_11",2,224)]
def mot(p):
    d=collections.defaultdict(dict)
    for l in open(p):
        a=l.split(",")
        if len(a)>=6: d[int(float(a[0]))][int(float(a[1]))]=tuple(map(float,a[2:6]))
    return d
def crop(seq,f,b,T=128):
    im=cv2.imread(f"{ET}/all_dataset_view/all/{seq}/img1/{f:06d}.jpg");x,y,w,h=b;m=.2
    c=im[max(0,int(y-h*m)):int(y+h*(1+m)),max(0,int(x-w*m)):int(x+w*(1+m))]
    return cv2.resize(c,(T,T))
def tile(c,t,col):
    c=cv2.copyMakeBorder(c,24,0,0,0,cv2.BORDER_CONSTANT,value=(30,30,30));cv2.rectangle(c,(0,24),(c.shape[1]-1,c.shape[0]-1),col,3)
    cv2.putText(c,t,(3,16),cv2.FONT_HERSHEY_SIMPLEX,.42,col,1);return c
def txt(w,t,col=(0,255,255),h=26):
    a=np.zeros((h,w,3),np.uint8);cv2.putText(a,t,(4,18),cv2.FONT_HERSHEY_SIMPLEX,.5,col,1);return a
for s,gid,fb in CASES:
    gt=mot(f"{ET}/all_dataset_view/all/{s}/gt/gt.txt");blocks=[]
    for t in("botsort","bytetrack"):
        G=json.load(open(f"{H}/c4_{t}/gap_outcomes.json"));g=[x for x in G if x["sequence"]==s and x["gt_identity_id"].endswith(f"__{gid}") and x["frame_before"]==fb][0]
        fa=g["frame_after"];pid=g["pred_id_after"]
        d=f"{H}/viz2/{t}/{s}";hit=sorted([f for f in os.listdir(d) if pid is not None and f"_id{pid}_" in f],key=lambda f:abs(int(re.search(r"_f(\d+)_",f).group(1))-fa))[:2]
        gb=tile(crop(s,fb,gt[fb][gid]),f"GT before f{fb}",(0,255,0));ga=tile(crop(s,fa,gt[fa][gid]),f"GT after f{fa}",(0,255,0))
        head=txt(1400,f"{t}  C4  outcome={g['outcome']}  gap={g['gap_len']}  pred_id before={g['pred_id_before']} after={pid}")
        rows=[]
        for f in hit:
            v=cv2.imread(f"{d}/{f}");v=cv2.resize(v,None,fx=1.5,fy=1.5,interpolation=cv2.INTER_CUBIC)
            rows.append(np.vstack([txt(v.shape[1],f"module: {f[:-4]}",(200,200,200),22),v]))
        if not rows: rows=[np.full((80,500,3),40,np.uint8)];cv2.putText(rows[0],"sin intento de reid cerca (id no nace/evaluable)",(6,45),cv2.FONT_HERSHEY_SIMPLEX,.5,(200,200,200),1)
        ctx=np.vstack([gb,np.zeros((8,gb.shape[1],3),np.uint8),ga])
        mod=np.vstack(sum([[r,np.zeros((6,r.shape[1],3),np.uint8)] for r in rows],[]))
        hh=max(ctx.shape[0],mod.shape[0]);W=ctx.shape[1]+10+mod.shape[1]
        body=np.zeros((hh,W,3),np.uint8);body[:ctx.shape[0],:ctx.shape[1]]=ctx;body[:mod.shape[0],ctx.shape[1]+10:]=mod
        Wm=max(W,1400);body=np.pad(body,((0,0),(0,Wm-W),(0,0)));head=np.pad(head,((0,0),(0,Wm-head.shape[1]),(0,0)));blocks.append(np.vstack([head,body]))
    W=max(b.shape[1] for b in blocks);blocks=[np.pad(b,((0,6),(0,W-b.shape[1]),(0,0))) for b in blocks]
    cv2.imwrite(f"{OUT}/{s[:14]}_gt{gid}_{fb}.jpg",np.vstack(blocks));print(s,"ok")
