"""
Build, from the published dataset, the image caches read by the training scripts.

The training scripts index the multi-view session by "proxy" frames: frame j of camera c is
source frame PROXY_START[c] + j of its original video (see timebase.py). The dataset videos are
resampled on the clock of iphone1, and `scene_mobile/frames.csv` gives, for every dataset
frame k, the proxy index of each camera. This script inverts that table to rebuild the caches
in the proxy indexing. For iphone1, 2 and 3 the mapping is a constant shift. For iphone0 nine
proxy frames were dropped by the resampling; they are filled with the nearest kept frame.

Outputs, in this folder:
  <cam>/mvn_960.bin, mvn_960_off.npy, mvn_960_meta.json, mvn_mask.png
        half-resolution frames, undistorted, centred on a 960 x 960 canvas, JPEG
        (train_dec3d_zoned.py --native 960)
  iphone1/frames_256.npy (+ _meta.npz)
        512 px window that follows the gripper, resized to 256 px (encoder input)
  runs/static_i3/sfm/images/f#####.jpg and .../ceiling/f#####.jpg
        key frames of the two static clips at 960 x 540 (train_static3dgs.py)

Usage:
    py prepare_data.py --data <dataset folder>
    py prepare_data.py --data <dataset folder> --only mvn crops static
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

import mv_views as mv

HERE = Path(__file__).parent.resolve()
CAMS = ['iphone0', 'iphone1', 'iphone2', 'iphone3']
# number of proxy frames per camera in the original layout (60 to 665 s of common time)
N_PROXY = {'iphone0': 36272, 'iphone1': 36263, 'iphone2': 36263, 'iphone3': 36263}


LIMIT = None      # --limit : only the first proxy frames (tests)


def proxy_to_k(data, cam):
    """Dataset frame k to read for each proxy frame j of `cam` (nearest kept frame if dropped)."""
    rows = list(csv.DictReader(open(Path(data) / 'scene_mobile' / 'frames.csv', encoding='utf-8')))
    have = {}
    for r in rows:
        have.setdefault(int(r['proxy_' + cam]), int(r['k']))
    js = np.array(sorted(have))
    out, n_fill = np.zeros(N_PROXY[cam], np.int64), 0
    for j in range(N_PROXY[cam]):
        if j in have:
            out[j] = have[j]
        else:
            out[j] = have[int(js[np.argmin(np.abs(js - j))])]
            n_fill += 1
    return (out[:LIMIT] if LIMIT else out), n_fill


def frames_in_order(path, ks):
    """Yield (j, BGR frame) for the dataset frames ks[j] (non-decreasing), reading once."""
    cap = cv2.VideoCapture(str(path))
    k_cur, im = -1, None
    for j, k in enumerate(ks):
        while k_cur < k:
            ok, im = cap.read()
            if not ok:
                raise RuntimeError('%s : no frame %d' % (path, k_cur + 1))
            k_cur += 1
        yield j, im
    cap.release()


def make_mvn(data, cam, canvas=960, quality=95):
    """Same as the original make_mv_native.py, with half-resolution frames of the dataset."""
    rig = mv.load_rig('rig_manual.json')
    if cam == 'iphone3':
        p0 = next(iter(json.loads((HERE / 'i3_poses_dense_s1.json').read_text(encoding='utf-8'))
                       ['iphone3_poses'].values()))
        K, dist, size = np.array(p0['K'], float), np.array(p0['dist'], float), p0['size']
    else:
        c = rig['cameras'][cam]
        K, dist, size = np.array(c['K'], float), np.array(c['dist'], float), c['size']
    Wp, Hp = int(size[0]) // 2, int(size[1]) // 2
    Kp = K.copy()
    Kp[:2] *= 0.5
    d5 = np.zeros(5)
    d5[:len(dist)] = dist
    mx, my = cv2.initUndistortRectifyMap(Kp, d5, None, Kp, (Wp, Hp), cv2.CV_16SC2)
    ok = cv2.remap(np.full((Hp, Wp), 255, np.uint8), mx, my, cv2.INTER_NEAREST,
                   borderMode=cv2.BORDER_CONSTANT, borderValue=0) == 255
    C = canvas
    x0, y0 = (C - Wp) // 2, (C - Hp) // 2
    mask = np.zeros((C, C), np.uint8)
    mask[y0:y0 + Hp, x0:x0 + Wp] = ok.astype(np.uint8) * 255
    (HERE / cam).mkdir(exist_ok=True)
    cv2.imwrite(str(HERE / cam / 'mvn_mask.png'), mask)
    Kc = Kp.copy()
    Kc[0, 2] += x0
    Kc[1, 2] += y0
    ks, n_fill = proxy_to_k(data, cam)
    off = [0]
    canvas_im = np.zeros((C, C, 3), np.uint8)
    with open(HERE / cam / ('mvn_%d.bin' % C), 'wb') as out:
        for j, im in frames_in_order(Path(data) / 'scene_mobile' / (cam + '.mp4'), ks):
            assert im.shape[:2] == (int(size[1]), int(size[0])), (cam, im.shape, size)
            half = cv2.resize(im, (Wp, Hp), interpolation=cv2.INTER_AREA)
            canvas_im[y0:y0 + Hp, x0:x0 + Wp] = cv2.remap(half, mx, my, cv2.INTER_LINEAR,
                                                         borderMode=cv2.BORDER_CONSTANT)
            buf = cv2.imencode('.jpg', canvas_im, [cv2.IMWRITE_JPEG_QUALITY, quality])[1]
            out.write(buf.tobytes())
            off.append(off[-1] + len(buf))
            if (j + 1) % 5000 == 0:
                print('%s : %d / %d frames, %.1f GB' % (cam, j + 1, len(ks), off[-1] / 1e9),
                      flush=True)
    np.save(HERE / cam / ('mvn_%d_off.npy' % C), np.array(off, np.int64))
    (HERE / cam / ('mvn_%d_meta.json' % C)).write_text(json.dumps(dict(
        K=Kc.tolist(), proxy_size=[Wp, Hp], offset=[x0, y0], canvas=C, frames=len(ks),
        quality=quality, note='undistorted frames centred on the canvas; K of the canvas'),
        indent=1), encoding='utf-8')
    print('%s : %d frames (%d filled from a neighbour) -> %.1f GB'
          % (cam, len(ks), n_fill, off[-1] / 1e9), flush=True)


def make_crops(data, win=512, dy=180, side=256):
    """Same as the original make_object_crops.py --cam iphone1 --win 512 --dy 180, written
    directly as the uint8 memmap of make_frames_memmap.py (RGB, (T, 3, side, side))."""
    cam = 'iphone1'
    g = np.load(HERE / cam / 'gripper.npy')[:, :2].astype(np.float64) * (1080.0 / 540)
    ks, _ = proxy_to_k(data, cam)
    T = min(len(g), len(ks))
    out = HERE / cam / ('frames_%d.npy' % side)
    mm = np.lib.format.open_memmap(out, mode='w+', dtype=np.uint8, shape=(T, 3, side, side))
    pad, sc, h = win, side / float(win), win // 2
    for j, img in frames_in_order(Path(data) / 'scene_mobile' / (cam + '.mp4'), ks[:T]):
        big = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        fx, fy = g[j, 0] + pad, g[j, 1] + dy + pad
        cx, cy = int(np.floor(fx)), int(np.floor(fy))
        o = cv2.resize(big[cy - h:cy + h, cx - h:cx + h], (side, side),
                       interpolation=cv2.INTER_AREA)
        M = np.array([[1.0, 0.0, -(fx - cx) * sc], [0.0, 1.0, -(fy - cy) * sc]])
        o = cv2.warpAffine(o, M, (side, side), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
        mm[j] = cv2.cvtColor(o, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        if (j + 1) % 5000 == 0:
            print('crops : %d / %d' % (j + 1, T), flush=True)
    mm.flush()
    np.savez(out.with_name(out.stem + '_meta.npz'), n=T, side=side, video='dataset')
    print('crops : %d frames -> %s' % (T, out), flush=True)


def make_static(data):
    """Key frames of the static clips, as written by sfm_static_i3.py / sfm_add_video.py."""
    P = json.loads((HERE / 'runs/static_i3/poses_room.json').read_text(encoding='utf-8'))
    W, H = P['size']
    root = HERE / P['images_dir']
    vids = {'base': 'iphone3_fixed_scene.mp4', 'ceiling': 'iphone3_fixed_scene_2.mp4'}
    for v, fn in vids.items():
        want = sorted({int(e['frame']): name for name, e in P['poses'].items()
                       if e.get('video', 'base') == v}.items())
        ks = [f for f, _ in want]
        for j, im in frames_in_order(Path(data) / 'scene_statique' / fn, ks):
            dst = root / want[j][1]
            dst.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst), cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
        print('static %s : %d key frames' % (v, len(ks)), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='dataset folder (scene_mobile/, scene_statique/)')
    p.add_argument('--only', nargs='*', default=['mvn', 'crops', 'static'])
    p.add_argument('--cams', nargs='*', default=CAMS)
    p.add_argument('--limit', type=int, default=0, help='first proxy frames only (tests)')
    a = p.parse_args()
    global LIMIT
    LIMIT = a.limit or None
    if 'static' in a.only:
        make_static(a.data)
    if 'crops' in a.only:
        make_crops(a.data)
    if 'mvn' in a.only:
        for cam in a.cams:
            make_mvn(a.data, cam)


if __name__ == '__main__':
    main()
