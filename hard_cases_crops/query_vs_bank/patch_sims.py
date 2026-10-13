ET="/mnt/Disk2/RT-DETR_4/RT-DETR/rtdetrv2_pytorch/eval_tracking"
s=open(f"{ET}/main_track_viz.py").read()
a="""                                    viz_top_k=args.reid_stitch_viz_top_k)
"""
assert s.count(a)==1
s=s.replace(a,a+"""        import os as _os, json as _json
        if _os.environ.get("SIMLOG"):  # SIMLOG: registra sims contra TODO el banco en cada intento
            _orig_mn = reid_gallery.match_newborns
            def _mn(embeds, crops=None, frame_id=None, raw_ids=None):
                lost = list(reid_gallery._bank.keys())
                if lost and len(embeds):
                    sims = np.stack([(embeds @ reid_gallery._bank[f]["embeds"].T).mean(axis=1) for f in lost], axis=1)
                else:
                    sims = np.zeros((len(embeds), 0))
                res = _orig_mn(embeds, crops=crops, frame_id=frame_id, raw_ids=raw_ids)
                with open(_os.environ["SIMLOG"], "a") as _f:
                    _f.write(_json.dumps({"frame": frame_id, "raw_ids": raw_ids, "lost": lost,
                                          "sims": sims.round(4).tolist(), "res": res}) + "\\n")
                return res
            reid_gallery.match_newborns = _mn
""")
open(f"{ET}/main_track_viz.py","w").write(s);print("ok")
