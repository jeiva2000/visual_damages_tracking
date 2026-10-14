# exps/

Scripts of the experiments run on the GPU server, one folder per experiment:

| Folder | Experiment |
|---|---|
| `exp_exclude_seen/` | Recomputation of every result without the 10 sequences whose frames the detector saw (live, k-fold, ablations, McNemar) |
| `exp_novideo_detector/`, `exp_noglass_detector/`, `exp_custommot2_detector/` | Alternative detectors trained without the video frames |
| `exp_testmodels/` | Native-tracker sweep over the detectors in `test_models` |
| `exp_video_ablation/` | Detector with and without the video frames |

These scripts contain the **absolute paths of the server** where they were run (`/mnt/Disk2/...`), and they read and
write data (caches, run outputs) that is not part of the repository. They document exactly how each reported
number was obtained; to rerun them, adapt those paths. The commands of the main live evaluation, portable, are in
`../live_tracking/run_live_experiments.sh`.
