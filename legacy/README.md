# legacy/

Scripts from earlier stages of the project that are **not needed to reproduce the paper**. They are kept for
history only and were moved here because nothing in the main code imports them. They may not run from this
folder as is (some import modules that stay in the repository root, e.g. `gallery_from_live_preds`; run them
with `PYTHONPATH=..`).

| File | What it was |
|---|---|
| `find_live_examples.py`, `native_vs_c4_compose.py`, `native_vs_c4_more.py`, `gallery_viz_template.py`, `gallery_success2.py` | Early search and composition of qualitative examples (superseded by `paper_fig_qualitative_native_vs_c4_v2.py`) |
| `print_sims_*.py`, `visualize_c4_proof.py`, `visualize_case_diffs.py`, `scan_candidates.py` | One-off diagnostics of similarities and case differences |
| `run_encoder_sweep.sh` | Early encoder (DINOv2/DINOv3) sweep |
| `local_backups/` | Local `.bak` copies of three modules (not tracked by git) |

The full chronological record is in `../docs/EXPERIMENT_LOG.md`.
