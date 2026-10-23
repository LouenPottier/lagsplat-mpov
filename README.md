# lagsplat-mpov

Training code for the 3D experiment on the **mpov** dataset: four synchronized smartphone
videos of a robot-held flexible strip, with camera poses. The method extends
[LaGSplat](https://arxiv.org/abs/2608.16324) (code of the paper:
<https://github.com/LouenPottier/LaGSplat>) to real multi-view footage.

- Dataset: [10.5281/zenodo.23601189](https://doi.org/10.5281/zenodo.23601189) (videos,
  camera poses, zones, measured input `s(t)`).
- Checkpoints of the published model: `checkpoints.zip` in the same Zenodo record.
- Web demo: <https://louenpottier.github.io/xrdays.html>.

This is the research code that produced the published model, with the paths adapted to the
dataset. A new training run gives a similar model, not the exact same numbers: the image caches
are rebuilt from the published videos, and the published decoder was obtained by a chain of
resumed runs and a final manual cleaning (see below).

## Installation

Python 3.13, a CUDA GPU (training and demo use [gsplat](https://github.com/nerfstudio-project/gsplat)).

```bash
pip install -r requirements.txt
```

## Using the published model

```bash
cd mpov
unzip ../checkpoints.zip            # -> runs/...
py demo3d.py --run hd               # local 3D demo, http://127.0.0.1:8783
```

`demo3d.py` uses the linear dynamics (`runs/dyn_lin`); the web demo uses the Lagrangian with an
invex potential (`runs/lnn_invex`).

`runs/hd/dec.pt` is the raw checkpoint, as trained. The PSNR given in the dataset note is
measured on it with

```bash
py eval_psnr_hd.py --ckpt runs/hd/dec.pt
```

(every held-out instant of the last 60 s with a pose of iphone3, 4 views at 960 px, whole
image and projection of the zone q; the 77 held-out images of the static scene; about 5 min). It needs the caches of `prepare_data.py`.

## Training from the dataset

```bash
cd mpov
bash run_all.sh <dataset folder>          # all stages
bash run_all.sh <dataset folder> 2 3      # stages 2 to 3 only
```

| stage | script | output |
|---|---|---|
| 0 | `prepare_data.py` | image caches rebuilt from the dataset videos (CPU) |
| 1 | `train_static3dgs.py` | static 3DGS of the room, `runs/static3dgs_v4/scene.pt` |
| 2 | `train_ae_demoxr.py`, `freeze_encoder.py` | encoder on iphone1, `runs/i1_d2_frozen/ae.pt` |
| 3 | `encode_z.py` | state `z(t) = (s, q0, q1)`, `runs/z_all_scorr.npy` |
| 4 | `fit_dynamics_mpov.py`, `fit_lnn_invex.py` | dynamics, `runs/dyn_lin`, `runs/lnn_invex` |
| 5 | `train_dec3d_zoned.py` | 3D decoder by zones, `runs/dec3d_zoned/dec_best.pt` |

### Manual cleaning

For the web demo, floating, wrong or low-opacity Gaussians were removed by hand with
`prune_tool.py` (lasso in a rendered view, or one Gaussian at a time), then applied with
`apply_prune_zoned.py`. On the published checkpoint this removed 946 Gaussians; the zones and
indices are in `runs/hd/dec_prune_regions.json` and `runs/hd/dec_prune_ids.json`, and only apply
to that checkpoint:

```bash
py apply_prune_zoned.py --ckpt runs/hd/dec.pt          # -> runs/hd/dec_pruned.pt
```

This cleaning lowers the PSNR of the whole images (the removed Gaussians still helped to
render parts of the background) and leaves the PSNR on the strip almost unchanged. To clean a
new training run:

```bash
py prune_tool.py --run dec3d_zoned --ckpt runs/dec3d_zoned/dec_best.pt
py apply_prune_zoned.py --ckpt runs/dec3d_zoned/dec_best.pt
```

## Inputs shipped with the code

The scripts read the geometry in its original format (the dataset gives the same information
indexed on the dataset frames):

| file | content |
|---|---|
| `rig_manual.json` | intrinsics and poses of the fixed cameras, manually clicked points |
| `i3_poses_dense_s1.json` | pose of the handheld iphone3 per frame |
| `drift.json`, `timebase.py` | clock model of the four cameras (frame indexing of the caches) |
| `iphone0/gripper.npy`, `iphone1/gripper.npy` | gripper track in each view |
| `iphone1/s_track.npy` | measured input `s(t)` (arc length of the gripper path) |
| `runs/static_i3/*` | poses of the static clips, SfM points, corrected and excluded poses, zones |
| `inputs/all_corr.npz` | per-frame corrections of the iphone3 poses and of `s` |
| `runs/dec3d_shared/*` | vertical direction and planes used by the demo and the cleaning tool |

`inputs/all_corr.npz` comes from a registration of the iphone3 frames against an earlier
decoder (`refine_worst.py` in the original code, not included); it is shipped as data.

## Licence

Code, dataset and checkpoints:
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (attribution, non-commercial
use only). For any commercial use, contact the author.

## Citation

Please cite the dataset note (arXiv link to come) and LaGSplat:

```bibtex
@misc{pottier2026lagsplat,
  title  = {LaGSplat: Inferring Physics-Governed Interactive Simulation from Monocular
            Video Using Latent Lagrangian Gaussian Splatting},
  author = {Pottier, Louen},
  year   = {2026},
  eprint = {2608.16324},
  archivePrefix = {arXiv}
}
```
