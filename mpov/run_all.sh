#!/usr/bin/env bash
# Full training recipe of the mpov case, from the published dataset.
#
#   bash run_all.sh <dataset folder> [first stage] [last stage]
#
# Stages: 0 caches, 1 static scene, 2 encoder, 3 state z(t), 4 dynamics, 5 decoder.
# Every stage writes into runs/ and reads the outputs of the previous ones.
# Indicative times on one RTX 4070 Ti: 0 about 1 h (CPU), 1 about 1 h, 2 about 2 h,
# 3 a few minutes, 4 about 30 min, 5 about 2 days for 1 000 000 steps.
set -euo pipefail
cd "$(dirname "$0")"
DATA=${1:?usage: bash run_all.sh <dataset folder> [first] [last]}
FIRST=${2:-0}
LAST=${3:-5}
run() { [ "$1" -ge "$FIRST" ] && [ "$1" -le "$LAST" ]; }

if run 0; then
  echo "== 0. image caches from the dataset (CPU)"
  py -u prepare_data.py --data "$DATA"
fi

if run 1; then
  echo "== 1. static 3DGS of the room"
  py -u train_static3dgs.py --out static3dgs_v4 --steps 30000 --n-init 100000 --max-add 50000 \
    --sh-degree 0 --cam-clear 0.3 --near 0.2 --fixed-poses runs/static_i3/fixed_poses.json \
    --exclude base:2510-2550 base:5505-5520 base:6100-6215 base:6260-6303
fi

if run 2; then
  echo "== 2. encoder on iphone1, then frozen"
  COMMON="--frames iphone1/frames_256.npy --masks none --track iphone1/s_track.npy \
    --u-sigma 0 --d-q 2 --w-kl 0 --val-sec 60 --plot-every 500 --gaussians 20000"
  py -u train_ae_demoxr.py $COMMON --steps 5000 --n-win 16 --n-frame 8 --out i1_d2
  py -u train_ae_demoxr.py $COMMON --steps 1000 --n-win 64 --micro-win 16 --n-frame 8 \
    --lr-enc 3e-5 --init runs/i1_d2/ae.pt --out i1_d2_b512
  py -u freeze_encoder.py --ckpt runs/i1_d2_b512/ae.pt --frames iphone1/frames_256.npy \
    --track iphone1/s_track.npy --u-sigma 0 --out runs/i1_d2_frozen/ae.pt
fi

if run 3; then
  echo "== 3. state z(t) = (s, q0, q1), with the refined s"
  py -u encode_z.py --ae runs/i1_d2_frozen/ae.pt --s-corr inputs/all_corr.npz \
    --out runs/z_all_scorr.npy
fi

if run 4; then
  echo "== 4. dynamics: linear fit, then Lagrangian with an invex potential"
  py -u fit_dynamics_mpov.py --z runs/z_all_scorr.npy --out dyn_lin
  py -u fit_lnn_invex.py --z runs/z_all_scorr.npy --init-linear runs/dyn_lin/model.pt \
    --out lnn_invex
fi

if run 5; then
  echo "== 5. 3D decoder by zones (settings of the published checkpoint)"
  py -u train_dec3d_zoned.py --out dec3d_zoned --scene runs/static3dgs_v4/scene.pt \
    --z-all runs/z_all_scorr.npy --i3-pose-corr inputs/all_corr.npz \
    --i3-exclude inputs/all_corr_exclude.npy --global-sim 1 \
    --exact 1 --native 960 --color-sigmoid 1 --carrier-src 12 --carrier-deg 6 \
    --prune-small-px 8.5 --w-q-edge 3 --w-s-far 1.5 --fast-v0 200 --fast-min 0.2 \
    --learn-s-fixed 1 --learn-poses 1 --exclude-poses none --q-free-trans 1.5 \
    --q-gate-power 2 --w-smooth 10 --smooth-edges 32768 --w-aniso 1e-2 --cov-eps 0.0009 \
    --guard 2 --snapshot-every 100000 --steps 1000000
fi
echo "done"
