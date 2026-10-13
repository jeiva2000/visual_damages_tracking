import re
ET="/mnt/Disk2/RT-DETR_4/RT-DETR/rtdetrv2_pytorch/eval_tracking"
s=open(f"{ET}/main_track_nodedup.py").read()
a="""                    for row, scale_row in zip(tracks, all_scale_embeds):
                        track_history_c4.setdefault(int(row[4]), []).append(scale_row)
"""
assert s.count(a)==1
s=s.replace(a,a+"""                    if args.reid_stitch_viz_dir:  # VIZ: crops alineados 1-a-1 con track_history_c4
                        for row in tracks:
                            _c = _crop_with_margin(img, tuple(row[:4]), args.reid_margin_ratio)
                            _viz_crops_c4.setdefault(int(row[4]), []).append(_c)
""")
b="""                elif c4_stitch:
                    history = track_history_c4.pop(raw_id, None)
                    embeds = reid.pool_history(history) if history else None
                    if embeds is None:
                        continue
"""
assert s.count(b)==1
s=s.replace(b,b+"""                    if args.reid_stitch_viz_dir:  # VIZ: mismos frames que usa pool_history
                        _vc = _viz_crops_c4.pop(raw_id, None)
                        if _vc:
                            from custom_reid import _spread_indices_c4
                            crop = [_vc[i] for i in _spread_indices_c4(len(_vc), reid.num_track_history) if _vc[i] is not None and _vc[i].size]
""")
c="    last_crop = {}\n"
assert s.count(c)==1
s=s.replace(c,c+"    _viz_crops_c4 = {}\n")
open(f"{ET}/main_track_viz.py","w").write(s)
r=open(f"{ET}/run_eval_nodedup.py").read()
r=r.replace("import main_track_nodedup as main_track","import main_track_viz as main_track")
open(f"{ET}/run_eval_viz.py","w").write(r)
print("ok")
