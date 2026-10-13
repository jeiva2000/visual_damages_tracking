import json,os,cv2,numpy as np,collections,sys
ET="/mnt/Disk2/RT-DETR_4/RT-DETR/rtdetrv2_pytorch/eval_tracking"
OUT=f"{ET}/exp_exclude_seen/hard_cases/crops"; os.makedirs(OUT,exist_ok=True)
CASES=[("video_75wAEktItLIYWrWOW7E_11",2,183),("video_0vbcxRjnfa70hc8AVvt_11",1,170),("video_0XWkyECiQbjitGSpVQK_11",4,292),
("video_Ve7DX8i8A6RX5h8OyFt_11",5,31),("video_0jGlrs0wJs3utJftEoD_11",2,289),("video_0jGlrs0wJs3utJftEoD_11",2,347),
("video_1Vl7JUY8fBQhAph15I7_11",1,19),("video_rICjFXthyXrdrgr58oE_11",1,169),
("video_kYUj2rFivaSQMwnjjGp_11",2,234),("video_Mpn2iv5EwC8BQM4Guf2_11",1,78),("video_Mpn2iv5EwC8BQM4Guf2_11",2,224)]
def load_mot(p):
    d=collections.defaultdict(dict)
    for l in open(p):
        a=l.strip().split(",")
        if len(a)<6: continue
        d[int(float(a[0]))][int(float(a[1]))]=tuple(map(float,a[2:6]))
    return d
def iou(a,b):
    ax2,ay2,bx2,by2=a[0]+a[2],a[1]+a[3],b[0]+b[2],b[1]+b[3]
    iw=max(0,min(ax2,bx2)-max(a[0],b[0]));ih=max(0,min(ay2,by2)-max(a[1],b[1]));i=iw*ih
    u=a[2]*a[3]+b[2]*b[3]-i;return i/u if u>0 else 0
def img(seq,f):
    return cv2.imread(f"{ET}/all_dataset_view/all/{seq}/img1/{f:06d}.jpg")
def crop(im,b,pad=0.15):
    x,y,w,h=b;H,W=im.shape[:2];px,py=w*pad,h*pad
    c=im[max(0,int(y-py)):min(H,int(y+h+py)),max(0,int(x-px)):min(W,int(x+w+px))]
    return c
def fit(c,h=260):
    if c is None or c.size==0: return np.full((h,200,3),60,np.uint8)
    return cv2.resize(c,(max(1,int(c.shape[1]*h/c.shape[0])),h))
def label(c,t):
    c=c.copy();cv2.rectangle(c,(0,0),(c.shape[1],22),(0,0,0),-1);cv2.putText(c,t,(4,16),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1);return c
def draw(im,b,col,t):
    im=im.copy();x,y,w,h=map(int,b);cv2.rectangle(im,(x,y),(x+w,y+h),col,3);cv2.putText(im,t,(x,max(15,y-5)),cv2.FONT_HERSHEY_SIMPLEX,.8,col,2);return im
rows=[]
for seq,gid,fb in CASES:
    gt=load_mot(f"{ET}/all_dataset_view/all/{seq}/gt/gt.txt")
    res={}
    for m in ("c4","p4fpn"):
        for t in ("botsort","bytetrack","ocsort","deepocsort"):
            g=[x for x in json.load(open(f"{ET}/exp_exclude_seen/hard_cases/{m}_{t}/gap_outcomes.json")) if x["sequence"]==seq and x["gt_identity_id"].endswith(f"__{gid}") and x["frame_before"]==fb]
            if g: res[(m,t)]=g[0]
    g0=next(iter(res.values())); fa=g0["frame_after"]
    tag=" ".join(f"{m}/{t[:4]}:{r['outcome'][:4]}" for (m,t),r in res.items() if r['outcome']!='unevaluated')
    gtb=gt[fb][gid]; gta=gt[fa][gid]
    parts=[]
    imb,ima=img(seq,fb),img(seq,fa)
    parts.append(label(fit(draw(imb,gtb,(0,255,0),f"GT{gid}"),260),f"frame {fb} (before)"))
    parts.append(label(fit(draw(ima,gta,(0,255,0),f"GT{gid}"),260),f"frame {fa} (after, gap={g0['gap_len']})"))
    parts.append(label(fit(crop(imb,gtb),260),f"GT{gid} before"));parts.append(label(fit(crop(ima,gta),260),f"GT{gid} after"))
    # confuser: tracker(c4) where swap
    sw=[(k,r) for k,r in res.items() if r["outcome"]=="id_swap" and k[0]=="c4"]
    if sw:
        (m,t),r=sw[0];pred=load_mot(f"{ET}/exp_live_excl/{m}_live_{t}_alldata/txt/{seq}.txt")
        pid=r["pred_id_after"];cnt=collections.Counter();last={}
        for f,d in pred.items():
            if pid in d and f<fa:
                for ogid,b in gt.get(f,{}).items():
                    if ogid!=gid and iou(d[pid],b)>=.5: cnt[ogid]+=1;last[ogid]=(f,b)
        if cnt:
            og=cnt.most_common(1)[0][0];f,b=last[og]
            parts.append(label(fit(crop(img(seq,f),b),260),f"confuser GT{og} (f{f}, {t[:4]})"))
        else: parts.append(label(fit(None),f"pred id {pid} no GT match"))
    h=max(p.shape[0] for p in parts);row=np.hstack([np.pad(p,((0,h-p.shape[0]),(0,0),(0,0))) for p in parts])
    hdr=np.zeros((28,row.shape[1],3),np.uint8);cv2.putText(hdr,f"{seq[:24]} gt{gid} {fb}->{fa} | {tag}",(4,20),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,255),1)
    row=np.vstack([hdr,row]);cv2.imwrite(f"{OUT}/{seq[:14]}_gt{gid}_{fb}.jpg",row);print(seq,gid,fb,"ok")
