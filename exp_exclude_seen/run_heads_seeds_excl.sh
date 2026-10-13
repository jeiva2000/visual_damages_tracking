#!/usr/bin/env bash
# Cola: H1 (v6style) y H4 (pure_triplet) sobre C4, EXCL, con 2 semillas de torch
# más (1, 2), para medir cuánto de la ventaja de las cabezas en OC-SORT/DeepOC-SORT
# es ruido de inicialización. La corrida original EXCL (sin semilla) es la 3ª muestra.
cd "$(dirname "$0")"
for s in 1 2; do
  HEAD_TORCH_SEED=$s VARIANTS="v6style pure_triplet" HGROUPS="c4_btbs c4_ocdoc" ./run_heads_excl.sh || exit 1
done
echo DONE > heads_seeds_EXCL.done
