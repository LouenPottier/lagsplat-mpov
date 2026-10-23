"""
PSNR du decodeur par zones sur un SOUS-ECHANTILLON du held-out, a faible cout GPU.

Seance multivue : `--n` instants repartis sur les `--val-sec` dernieres secondes (held-out de
l'entrainement), rendus UN par un, 4 vues sur le carre natif 960. Meme construction des vues que
`train_dec3d_zoned.py` (split 'val' : poses fixes corrigees + similitude, iphone3 non corrige,
masque de l'iphone0 aux instants rapides, correction de couleur color1). Deux PSNR par vue :
  - image entiere (pixels valides du carre) ;
  - zone q seule : enveloppe convexe du pave q, translate par le porteur, projetee (la languette
    et ce qui l'entoure).
Scene statique : `--n-static` images held-out, rendues a (s_fixed, 0, 0), PSNR brut et apres
gain/biais par canal ajuste sur l'image (comme l'evaluation de l'entrainement).

Geometrie lue dans `--ckpt` ; correction de couleur color1 lue dans `--color-ckpt` (le
checkpoint `_lowcol` a remplace celles de l'iphone1 et de l'iphone3 pour l'affichage web).

    py eval_psnr_hd.py
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch

import mv_views as mv
import timebase
from train_dec3d_zoned import ZonedSplat, scene_from_ckpt, JpegBlob, shown_proxy, proxy_to_ref

HERE = Path(__file__).parent.resolve()


def se3(xi):
    w, t = xi[:, :3], xi[:, 3:]
    th = w.norm(dim=1, keepdim=True).clamp(min=1e-9)
    k = w / th
    Kx = torch.zeros(len(xi), 3, 3, device=xi.device)
    Kx[:, 0, 1], Kx[:, 0, 2], Kx[:, 1, 2] = -k[:, 2], k[:, 1], -k[:, 0]
    Kx = Kx - Kx.transpose(1, 2)
    th = th[:, :, None]
    R = torch.eye(3, device=xi.device) + torch.sin(th) * Kx + (1 - torch.cos(th)) * Kx @ Kx
    M = torch.eye(4, device=xi.device).repeat(len(xi), 1, 1)
    M[:, :3, :3], M[:, :3, 3] = R, t
    return M


def psnr(x, y, m):
    """x, y (3, H, W) dans [0, 1], m (H, W) booleen."""
    mse = float(((x - y) ** 2).mean(0)[m].mean())
    return -10 * math.log10(max(mse, 1e-12))


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', default='runs/hd/dec.pt')
    p.add_argument('--color-ckpt', default='runs/hd/dec.pt')
    p.add_argument('--n', type=int, default=0, help='instants held-out evalues (0 = tous)')
    p.add_argument('--n-static', type=int, default=0, help='images statiques held-out (0 = toutes)')
    p.add_argument('--out', default='runs/eval_psnr_hd.json')
    p.add_argument('--sheet', default='', help='png : cible / rendu du premier instant et de la premiere image statique')
    a = p.parse_args()
    dev = 'cuda'
    ck = torch.load(HERE / a.ckpt, map_location=dev, weights_only=False)
    cc = torch.load(HERE / a.color_ckpt, map_location='cpu', weights_only=False)['color1']
    A = ck['args']
    d = ck['dec']
    sc = torch.load(HERE / A['scene'], map_location='cpu', weights_only=False)
    scg = scene_from_ckpt(d, sc['sh_degree'])
    dec = ZonedSplat(scg, d['idx_s'].long(), d['idx_q'].long(), d['z_star'], A['hidden'],
                     A['layers'], A['max_dxyz'], d['carrier_coef'].cpu().numpy(),
                     d['carrier_rng'].cpu().numpy(), d['carrier_w']).to(dev)
    dec.load_state_dict(d, strict=False)
    dec.color_q = bool(A.get('color_q', 1))
    dec.eval()
    near, C = A['near'], A['native']
    zones = json.loads((HERE / A['zones']).read_text(encoding='utf-8'))

    # ── vues de la seance (comme train_dec3d_zoned.py, split 'val') ─────────────────────
    rig = mv.load_rig(A['rig'])
    views = mv.build_views(rig, cams=list(mv.FIXED), side=C, full=True)
    frames = {v['cam']: JpegBlob(v['cam'], C) for v in views}
    m3 = JpegBlob('iphone3', C)
    K_fix = torch.tensor(np.stack([frames[v['cam']].meta['K'] for v in views]),
                         dtype=torch.float32, device=dev)
    E_fix = torch.tensor(np.stack([v['E'] for v in views]), dtype=torch.float32, device=dev)
    K3 = torch.tensor(m3.meta['K'], dtype=torch.float32, device=dev)
    Zall = np.load(HERE / A['z_all'])
    T = len(Zall)
    T_tr = T - int(A['val_sec'] * 60000 / 1001)
    g0 = np.load(HERE / 'iphone0' / 'gripper.npy')
    ok0 = mv.misreg_px('iphone0', np.arange(T, dtype=float),
                       np.abs(np.gradient(g0[:, 0]))) < A['i0_max_px']
    poses = json.loads((HERE / A['i3']).read_text(encoding='utf-8'))['iphone3_poses']
    bad = set(str(k) for k in np.load(HERE / A['i3_exclude'])) if A['i3_exclude'] else set()
    val = []
    for t, pz in poses.items():
        if t in bad:
            continue
        j3 = (int(pz['frame_i3']) - timebase.PROXY_START['iphone3'] if 'frame_i3' in pz
              else shown_proxy('iphone3', t))
        i_ref = proxy_to_ref('iphone3', j3)
        if T_tr <= i_ref < T and 0 <= j3 < len(m3):
            E = np.eye(4)
            E[:3, :3] = np.array(pz['R'], float)
            E[:3, 3] = np.array(pz['t'], float).ravel()
            val.append((int(round(i_ref)), j3, E))
    val.sort(key=lambda r: r[0])
    pick = val if a.n <= 0 else [val[i] for i in np.linspace(0, len(val) - 1, a.n).astype(int)]
    # poses fixes : similitude seance -> scene fixe, puis correction apprise
    R, s_ = dec.sim_rs()
    t_ = dec.world_sim[4:]
    Gi = torch.eye(4, device=dev)
    Gi[:3, :3], Gi[:3, 3] = R.T / s_, -(R.T @ t_) / s_
    Sd = torch.diag(torch.cat([s_.expand(3), torch.ones(1, device=dev)]))
    Ef = se3(ck['pose_fix'].to(dev)) @ (Sd @ E_fix @ Gi)
    ccA, ccb = cc['A'].to(dev), cc['b'].to(dev)
    masks = [torch.tensor(frames[v['cam']].mask, device=dev) for v in views] + \
        [torch.tensor(m3.mask, device=dev)]
    cams = [v['cam'] for v in views] + ['iphone3']
    Qc = torch.tensor(zones['zone_q']['corners'], dtype=torch.float32, device=dev)

    def hull(E4, K, shift):
        X = (Qc + shift) @ E4[:3, :3].T + E4[:3, 3]
        m_ = np.zeros((C, C), np.uint8)
        if bool((X[:, 2] > 0.05).all()):
            uv = (X[:, :2] / X[:, 2:3]) @ K[:2, :2].T + K[:2, 2]
            cv2.fillConvexPoly(m_, cv2.convexHull(uv.cpu().numpy().astype(np.float32))
                               .astype(np.int32), 1)
        return torch.from_numpy(m_ > 0).to(dev)

    sheet_rows = []
    full = {c: [] for c in cams}
    strip = {c: [] for c in cams}
    for i_ref, j3, E3 in pick:
        z = torch.tensor(Zall[i_ref][None], dtype=torch.float32, device=dev)
        vm = torch.cat([Ef, torch.tensor(E3, dtype=torch.float32, device=dev)[None]])[None]
        Km = torch.cat([K_fix, K3[None]])[None]
        out = dec.render(z, vm, Km, C, C, near)[0]                          # (V, 3, C, C)
        out = (torch.einsum('vij,vjhw->vihw', ccA, out) + ccb[:, :, None, None]).clamp(0, 1)
        sh = dec.carrier(z[:, 0]) - dec.carrier(dec.z_star[None, 0])
        for v, cam in enumerate(cams):
            if cam == 'iphone0' and not ok0[i_ref]:
                continue
            if cam == 'iphone3':
                tg = m3[j3]
            else:
                j = int(np.clip(round(float(mv.ref_to_proxy(cam, float(i_ref)))), 0,
                                len(frames[cam]) - 1))
                tg = frames[cam][j]
            tg = torch.tensor(np.asarray(tg, np.float32) / 255.0, device=dev)
            full[cam].append(psnr(out[v], tg, masks[v]))
            mq = hull(vm[0, v], Km[0, v], sh[0]) & masks[v]
            if int(mq.sum()) > 100:
                strip[cam].append(psnr(out[v], tg, mq))
        if a.sheet and not sheet_rows:
            for v, cam in enumerate(cams):
                tg_ = (m3[j3] if cam == 'iphone3' else frames[cam][int(np.clip(round(float(
                    mv.ref_to_proxy(cam, float(i_ref)))), 0, len(frames[cam]) - 1))])
                sheet_rows.append(np.concatenate([np.asarray(tg_).transpose(1, 2, 0),
                    (out[v].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)], 1))
        del out
        torch.cuda.empty_cache()

    # ── scene statique ──────────────────────────────────────────────────────────────
    P2 = json.loads((HERE / A['static_poses']).read_text(encoding='utf-8'))
    names2, is_val2 = list(sc['names']), np.asarray(sc['is_val'])
    W2, H2 = [int(v) for v in sc['size']]
    mx, my = cv2.initUndistortRectifyMap(np.array(P2['K'], float),
                                         np.r_[np.array(P2['dist'], float), 0.0], None,
                                         np.array(P2['K'], float), (W2, H2), cv2.CV_32FC1)
    valid2 = torch.tensor(cv2.remap(np.ones((H2, W2), np.float32), mx, my, cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0.999,
                          device=dev)
    VM2, K2 = sc['viewmats'].float().to(dev), sc['K'].float().to(dev)
    va2 = np.flatnonzero(is_val2)
    if a.n_static > 0:
        va2 = va2[np.linspace(0, len(va2) - 1, a.n_static).astype(int)]
    z2 = torch.tensor([[float(ck['s_fixed']), 0.0, 0.0]], device=dev)
    ps_raw, ps_fit = [], []
    for i in va2:
        im = cv2.imread(str(HERE / P2['images_dir'] / names2[i]))[:, :, ::-1]
        y = torch.tensor(cv2.remap(np.ascontiguousarray(im), mx, my, cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT), dtype=torch.float32,
                         device=dev).permute(2, 0, 1) / 255.0
        x = dec.render(z2, VM2[i][None, None], K2[None, None], W2, H2, near)[0, 0]
        ps_raw.append(psnr(x, y, valid2))
        if a.sheet and len(ps_raw) == 1:
            st = np.concatenate([(y.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8),
                                 (x.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)], 1)
            st = cv2.resize(st, (2 * C, int(round(2 * C * st.shape[0] / st.shape[1]))))
            sheet_rows.append(st)
        xf, yf = x[:, valid2], y[:, valid2]
        xm, ym = xf.mean(1, keepdim=True), yf.mean(1, keepdim=True)
        gg = ((xf - xm) * (yf - ym)).sum(1, keepdim=True) / ((xf - xm) ** 2).sum(1, keepdim=True)
        mse = float((((gg * (xf - xm) + ym).clamp(0, 1) - yf) ** 2).mean())
        ps_fit.append(-10 * math.log10(mse))

    if a.sheet:
        cv2.imwrite(str(HERE / a.sheet), cv2.cvtColor(np.concatenate(sheet_rows, 0),
                                                      cv2.COLOR_RGB2BGR)[::2, ::2])
    res = dict(ckpt=a.ckpt, color_ckpt=a.color_ckpt, n_instants=len(pick),
               psnr_full={c: float(np.mean(v)) for c, v in full.items()},
               psnr_strip={c: float(np.mean(v)) for c, v in strip.items()},
               n_full={c: len(v) for c, v in full.items()},
               n_strip={c: len(v) for c, v in strip.items()},
               static_n=len(va2), static_psnr_raw=float(np.mean(ps_raw)),
               static_psnr_gain_bias=float(np.mean(ps_fit)))
    (HERE / a.out).write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
