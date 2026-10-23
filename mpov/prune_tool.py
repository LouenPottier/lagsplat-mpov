"""
Outil de nettoyage a la main du decodeur 3D PARTAGE : on tourne autour de la scene, on dessine
une courbe fermee a l'ecran, et TOUT ce qui se trouve dans la partie du frustum de ce point de
vue qui se projette dans la courbe (sur toute la profondeur) est supprime et ajoute a
l'elagage de l'entrainement.

Pourquoi un serveur et non un viewer WebGL comme demo/3d.html du site : les gaussiennes ne
sont pas fixes, un MLP de ~290 M parametres les place a chaque etat. Le serveur evalue le
decodeur sur le GPU et rend l'image avec gsplat, exactement le rendu de l'entrainement ; la
page n'affiche que l'image et gere camera, lasso et curseur de s.

Les zones sont ecrites dans <run>/prune_regions.json (format de prune_regions.py). Le
MEME fichier est relu par train_dec3d_shared.py --prune-regions a chaque nettoyage.

Commandes de la page : celles du viewer demo/3d.html du site (zqsd, fleches, a/e, ijkl,
clic droit = orbite autour du centre, molette = zoom, maj+molette = pan, 0-9 et -/+ = vues,
V = vue dans l'URL), sauf le clic gauche, qui trace le lasso au lieu d'appliquer l'effort.
  curseurs s, q0, q1 : etat affiche (q en ecarts types, bouton « q median »)
  en mode lasso   dessiner la courbe, relacher -> apercu en VERT, Entree valide, Echap annule
  U               annule la derniere zone       R  recharge le dernier checkpoint

CHECKPOINT PAR ZONES (`--ckpt`, train_dec3d_zoned.py). Le decodeur est enveloppe comme dans
demo3d.py. Chaque zone garde l'etat (s, q) affiche quand on l'a validee, et les gaussiennes
exclues sont celles dont le centre est dans la zone A CET ETAT (meme test que
apply_prune_zoned.py, qui ecrit ensuite le checkpoint elague). Zones dans
<dossier>/<stem>_prune_regions.json ; pas de plan rouge ni de contraintes dans ce mode.
Mode CLIC (touche P ou bouton) : la gaussienne sous le curseur est surlignee en jaune, un clic
gauche la supprime. C'est celle qui contribue le plus au pixel dans le compositing (alpha x
transmittance, meme projection que gsplat). Indices dans <dossier>/<stem>_prune_ids.json.
U annule la derniere action (zone ou clic).

Usage :
    py prune_tool.py                       # -> http://127.0.0.1:8781
    py prune_tool.py --ckpt runs/dec3d_zoned_low20k_c05/dec.pt
    py prune_tool.py --run dec3d_shared_p1 --port 8781
"""
import argparse
import json
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import torch

import mv_views as mv
from prune_regions import load_regions, save_regions, inside_region, inside_regions
from constraints3d import load_items, save_items, inside_boxes, near_wall

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--run', default='dec3d_shared_p1')
    p.add_argument('--ckpt', default=None,
                   help='checkpoint ZonedSplat (train_dec3d_zoned.py) ; remplace --run')
    p.add_argument('--regions', default=None, help='defaut : <run>/prune_regions.json')
    p.add_argument('--floor', default='runs/dec3d_shared/floor_plane.npz',
                   help='donne la verticale (n = vers le bas) pour une orbite qui ne roule pas')
    p.add_argument('--constraints', default=None, help='defaut : <run>/constraints.json')
    p.add_argument('--plane', default='runs/dec3d_shared/red_plane_final.npz')
    p.add_argument('--plane-margin', type=float, default=0.15)
    p.add_argument('--panel', default='runs/dec3d_shared/red_plane_img.npz',
                   help='points du panneau rouge : centre et axes pour le dessiner')
    p.add_argument('--vertical', default='runs/dec3d_shared/vertical.json',
                   help="verticale de la scene, reglee et enregistree dans l'outil ; a defaut "
                        "celle des normales de disques (up_from_surfels.npy)")
    p.add_argument('--port', type=int, default=8781)
    p.add_argument('--no-open', action='store_true')
    return p.parse_args()


def scene_up(a):
    """Verticale de la scene (vers le haut), et d'ou elle vient. Ordre : fichier enregistre
    par l'outil, puis estimation par les normales des disques, puis le bord du panneau rouge
    (fausse de 15 degres, gardee en dernier recours)."""
    vf = HERE / a.vertical
    if vf.exists():
        up = np.asarray(json.loads(vf.read_text(encoding='utf-8'))['up'], float)
        src = str(vf.name)
    elif (HERE / 'runs/dec3d_shared/up_from_surfels.npy').exists():
        up = np.load(HERE / 'runs/dec3d_shared/up_from_surfels.npy').astype(float)
        src = 'normales des disques (up_from_surfels.npy)'
    else:
        up = -np.load(HERE / a.floor)['n'].astype(float)
        src = 'bord du panneau rouge (floor_plane.npz)'
    return up / np.linalg.norm(up), src


class Tool:
    def __init__(self, a):
        self.a = a
        self.lock = threading.Lock()
        self.zoned = a.ckpt is not None
        if self.zoned:
            from apply_prune_zoned import regions_path_of
            self.ckpt = HERE / a.ckpt
            self.run = self.ckpt.parent
            dflt = regions_path_of(self.ckpt)
        else:
            self.run = HERE / 'runs' / a.run
            self.ckpt = self.run / 'dec.pt'
            dflt = self.run / 'prune_regions.json'
        self.regions_path = Path(a.regions) if a.regions else dflt
        self.regions = load_regions(self.regions_path)
        self.cons_path = (Path(a.constraints) if a.constraints
                          else self.run / 'constraints.json')
        self.items = load_items(self.cons_path)
        pl = HERE / a.plane
        self.plane = np.load(pl) if pl.exists() else None
        if self.zoned:
            self.plane = None                # rien d'autre que les zones n'est applique ici
            self.cons_path = self.ckpt.with_name(self.ckpt.stem + '_constraints.json')
            self.items = []
            zf = HERE / torch.load(self.ckpt, map_location='cpu', weights_only=False)['args']['z_all']
        else:
            zf = self.run / 'z_all.npy'
        self.Zall = np.load(zf)
        self.sh_degree = None
        self.reg_mask = None
        self.history = []                    # 'zone' ou ('id', i), pour U
        if self.zoned:
            from apply_prune_zoned import ids_path_of, load_ids
            self.ids_path = ids_path_of(self.ckpt)
            self.ids = load_ids(self.ids_path)
        else:
            self.ids_path, self.ids = None, []
        self.load()

    def load(self):
        from models_deform import SharedDeformSplat3D
        for k in range(10):
            try:
                ck = torch.load(self.ckpt, map_location='cuda', weights_only=False)
                break
            except Exception:
                time.sleep(2)
        ar = ck['args']
        ae = torch.load(HERE / (ar.get('init_ae') or 'runs/i1_d2_frozen/ae.pt'),
                        map_location='cpu', weights_only=False)
        self.u_mean = float(np.atleast_1d(ae['u_mean'])[0])
        self.u_std = float(np.atleast_1d(ae['u_std'])[0])
        if self.zoned:
            from demo3d import load_decoder
            dec, deg = load_decoder(ck)
            with self.lock:
                self.dec, self.step, self.sh_degree = dec, int(ck['step']), deg
                self.cache, self.reg_mask = {}, None
            print('checkpoint par zones pas %d, %d gaussiennes' % (self.step, dec.n_gaussians),
                  flush=True)
            return
        dec = SharedDeformSplat3D(self.Zall.shape[1], ck['dec']['xyz'].shape[0],
                                  hidden=ar['hidden'], layers=ar['layers'],
                                  pe_cond=ar['pe_cond'], alpha_min=ar['alpha_min'],
                                  scale_min=ar['scale_min'], max_dxyz=ar['max_dxyz']).cuda()
        dec.load_state_dict(ck['dec'])
        dec.eval()
        with self.lock:
            self.dec, self.step = dec, int(ck['step'])
            self.cache = {}
        print('checkpoint pas %d, %d gaussiennes' % (self.step, dec.n_gaussians), flush=True)

    def z_of(self, s_raw, q=None):
        z = np.median(self.Zall, 0).astype(np.float64)
        if s_raw is not None:
            z[0] = (s_raw - self.u_mean) / self.u_std
        if q is not None:
            z[1:1 + len(q)] = q
        return z

    @torch.no_grad()
    def state(self, s_raw, q=None):
        key = (round(float(s_raw), 2),) + tuple(round(float(v), 3) for v in (q or ()))
        if key not in self.cache:
            if len(self.cache) > 8:
                self.cache.clear()
            zt = torch.tensor(self.z_of(s_raw, q)[None], dtype=torch.float32, device='cuda')
            xyz, quat, scale, alpha, color = self.dec.state(zt)
            st = dict(xyz=xyz[0], quat=quat[0], scale=scale[0], alpha=alpha, color=color[0])
            if self.zoned:
                # chaque zone a l'etat ou elle a ete tracee, comme apply_prune_zoned.py
                if self.reg_mask is None:
                    from apply_prune_zoned import zone_mask
                    self.reg_mask = zone_mask(self.dec, self.regions, self.z_of)
                    if self.ids:
                        self.reg_mask[torch.tensor(self.ids, device='cuda')] = True
                st['hidden'] = self.reg_mask
            else:
                st['hidden'] = inside_regions(st['xyz'], self.regions)
            self.cache[key] = st
        return self.cache[key]

    @torch.no_grad()
    def pick(self, st, keep, E, K, W, H, x, y):
        """Indice de la gaussienne (parmi `keep`) qui contribue le plus au point (x, y) du
        rendu : poids alpha_i T_i du compositing avant-arriere, calcule sur la projection de
        gsplat lui-meme (centres 2D, coniques, opacites, profondeurs, et donc meme elimination
        des gaussiennes non dessinees). -1 si rien ne couvre ce point."""
        from gsplat import rasterization
        idx = torch.nonzero(keep, as_tuple=True)[0]
        _, _, meta = rasterization(means=st['xyz'][idx], quats=st['quat'][idx],
                                   scales=st['scale'][idx],
                                   opacities=st['alpha'][idx].clamp(0, 0.99),
                                   colors=st['color'][idx], viewmats=E[None], Ks=K[None],
                                   width=W, height=H, near_plane=0.01, far_plane=100.0,
                                   sh_degree=self.sh_degree, packed=True)
        gid = meta['gaussian_ids']
        if gid is None or len(gid) == 0:
            return -1
        m2, con, op, dep = meta['means2d'], meta['conics'], meta['opacities'], meta['depths']
        dx, dy = x - m2[:, 0], y - m2[:, 1]
        pw = -0.5 * (con[:, 0] * dx * dx + con[:, 2] * dy * dy) - con[:, 1] * dx * dy
        a = (op * torch.exp(pw.clamp(max=0))).clamp(max=0.99)
        a = torch.where((a >= 1 / 255) & (pw <= 0), a, torch.zeros_like(a))
        o = torch.argsort(dep)
        a = a[o]
        T = torch.cumprod(torch.cat([torch.ones(1, device=a.device), 1 - a[:-1]]), 0)
        w = T * a
        j = int(torch.argmax(w))
        return int(idx[gid[o[j]]]) if float(w[j]) > 0 else -1

    def render(self, cam, s_raw, q=None, preview=None, show=None, hover=None):
        from gsplat import rasterization
        W, H = int(cam['W']), int(cam['H'])
        E = torch.tensor(np.array(cam['E'], np.float32).reshape(4, 4), device='cuda')
        f = float(cam['f'])
        K = torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], dtype=torch.float32,
                         device='cuda')
        t0 = time.time()
        with self.lock:
            st = self.state(s_raw, q)
            show = show or {}
            color = st['color'].clone()
            # exclusions : zones dessinees + paves + devant le plan rouge, comme l'entrainement
            excl = st['hidden'] | inside_boxes(st['xyz'], self.items)
            if self.plane is not None:
                n_ = torch.tensor(self.plane['n'], dtype=torch.float32, device='cuda')
                excl |= (st['xyz'] @ n_ + float(self.plane['d'])) > self.a.plane_margin
            n_excl = int(excl.sum())
            if show.get('excl'):
                keep = torch.ones_like(excl)
                self.paint(color, excl, [1.0, 0.1, 0.1], 0.65)
            else:
                keep = ~excl
            near = torch.zeros_like(excl)
            for it in self.items:
                if it['type'] == 'wall':
                    near |= near_wall(st['xyz'], it)
            n_wall = int((near & keep).sum())
            if show.get('walls'):
                self.paint(color, near, [0.1, 0.8, 0.9], 0.6)
            n_prev = 0
            if preview is not None:
                m = inside_region(st['xyz'], dict(E=E.tolist(), K=K.tolist(), poly=preview))
                m &= keep
                self.paint(color, m, [0.0, 1.0, 0.0], 1.0)
                n_prev = int(m.sum())
            hov = -1
            if hover is not None:
                hov = self.pick(st, ~excl, E, K, W, H, float(hover[0]), float(hover[1]))
                if hov >= 0:
                    hm = torch.zeros_like(keep)
                    hm[hov] = True
                    self.paint(color, hm, [1.0, 0.9, 0.0], 1.0)
            k = torch.nonzero(keep, as_tuple=True)[0]
            r, _, _ = rasterization(means=st['xyz'][k], quats=st['quat'][k],
                                    scales=st['scale'][k],
                                    opacities=st['alpha'][k].clamp(0, 0.99), colors=color[k],
                                    viewmats=E[None], Ks=K[None], width=W, height=H,
                                    near_plane=0.01, far_plane=100.0,
                                    sh_degree=self.sh_degree,
                                    render_mode='RGB')
            img = (r[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            n_hidden = n_excl
        ok, buf = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, 88])
        return buf.tobytes(), dict(ms=(time.time() - t0) * 1000, preview=n_prev,
                                   hidden=n_hidden, step=self.step,
                                   total=int(self.dec.n_gaussians),
                                   regions=len(self.regions), near_wall=n_wall,
                                   items=len(self.items), hover=hov,
                                   n_ids=len(self.ids))

    def pick_commit(self, cam, s_raw, q, x, y):
        """Supprime la gaussienne sous (x, y), celle surlignee au survol."""
        W, H, f = int(cam['W']), int(cam['H']), float(cam['f'])
        E = torch.tensor(np.array(cam['E'], np.float32).reshape(4, 4), device='cuda')
        K = torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], dtype=torch.float32,
                         device='cuda')
        from apply_prune_zoned import save_ids
        with self.lock:
            st = self.state(s_raw, q)
            i = self.pick(st, ~st['hidden'], E, K, W, H, float(x), float(y))
            if i >= 0:
                self.ids.append(i)
                self.history.append(('id', i))
                save_ids(self.ids_path, self.ids)
                self.cache, self.reg_mask = {}, None
        return i

    def paint(self, color, m, rgb, w):
        """color[m] <- (1 - w) color[m] + w rgb, en couleur directe ou en harmoniques (degre 0 :
        gsplat rend 0.5 + C0 sh0)."""
        t = torch.tensor(rgb, device=color.device, dtype=color.dtype)
        if self.sh_degree is None:
            color[m] = (1 - w) * color[m] + w * t
        else:
            color[m] = (1 - w) * color[m]
            color[m, 0] += w * (t - 0.5) / 0.28209479177387814

    def commit(self, cam, poly, s_raw=None, q=None):
        W, H, f = int(cam['W']), int(cam['H']), float(cam['f'])
        reg = dict(E=np.array(cam['E'], float).reshape(4, 4).tolist(),
                   K=[[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], poly=poly, near=0.01,
                   note='prune_tool, %s' % time.strftime('%Y-%m-%d %H:%M:%S'))
        if s_raw is not None:
            reg['state'] = dict(s=float(s_raw), q=[float(v) for v in (q or [])] or None)
        with self.lock:
            self.regions.append(reg)
            self.history.append('zone')
            save_regions(self.regions_path, self.regions)
            self.cache, self.reg_mask = {}, None
        print('zone %d ajoutee -> %s' % (len(self.regions), self.regions_path), flush=True)

    def set_items(self, items):
        with self.lock:
            self.items = items
            save_items(self.cons_path, items)

    def undo(self):
        with self.lock:
            if self.history and self.history[-1] != 'zone':
                from apply_prune_zoned import save_ids
                _, i = self.history.pop()
                if i in self.ids:
                    self.ids.reverse()
                    self.ids.remove(i)
                    self.ids.reverse()
                    save_ids(self.ids_path, self.ids)
                self.cache, self.reg_mask = {}, None
                return 'gaussienne %d remise' % i
            if self.history:
                self.history.pop()
            # copie horodatee AVANT de retirer : 61 zones ont deja ete perdues sans trace
            if self.regions:
                bak = self.regions_path.with_name('prune_regions_%s_%dzones.json'
                                                  % (time.strftime('%Y%m%d_%H%M%S'),
                                                     len(self.regions)))
                save_regions(bak, self.regions)
            if self.regions:
                self.regions.pop()
                save_regions(self.regions_path, self.regions)
            self.cache, self.reg_mask = {}, None
            return 'derniere zone annulee'

    def meta(self):
        rig = mv.load_rig('rig_manual.json')
        presets = []
        for c in ('iphone0', 'iphone1', 'iphone2'):
            E = np.eye(4)
            E[:3, :3] = np.array(rig['cameras'][c]['R'])
            E[:3, 3] = np.array(rig['cameras'][c]['t']).ravel()
            presets.append(dict(name=c, E=E.tolist()))
        # 7 poses d'iphone3 etalees dans le temps, pour completer les touches 0-9
        i3 = json.loads((HERE / 'i3_poses_dense_s1.json').read_text(encoding='utf-8'))
        i3 = i3['iphone3_poses']
        keys = sorted(i3, key=float)
        for k in np.linspace(0, len(keys) - 1, 7).astype(int):
            pz = i3[keys[k]]
            E = np.eye(4)
            E[:3, :3] = np.array(pz['R'])
            E[:3, 3] = np.array(pz['t']).ravel()
            presets.append(dict(name='iphone3 t=%s' % keys[k], E=E.tolist()))
        # pivot = centre de la scene (mediane des gaussiennes assez opaques) et rayon au
        # 90e centile, comme sceneBounds du viewer de reference
        with self.lock:
            st = self.state(float(np.median(self.Zall[:, 0])) * self.u_std + self.u_mean)
            xyz = st['xyz'][st['alpha'] > 0.2].cpu().numpy()
        piv = np.median(xyz, 0)
        rad = float(np.percentile(np.linalg.norm(xyz - piv, axis=1), 90))
        up, up_src = scene_up(self.a)
        down = (-up).tolist()
        s_all = self.Zall[:, 0] * self.u_std + self.u_mean
        qs = self.Zall[:, 1:]
        q_rng = [[float(np.percentile(qs[:, k], 0.5)), float(np.percentile(qs[:, k], 99.5)),
                  float(np.median(qs[:, k]))] for k in range(qs.shape[1])]
        red_n = (np.asarray(self.plane['n'], float) if self.plane is not None
                 else np.array([0.0, 0.0, -1.0]))
        panel = None
        pp = HERE / self.a.panel
        if pp.exists():
            zp = np.load(pp)
            P, inl = zp['P'][zp['inl']], zp['basis']
            c = P.mean(0)
            pr = (P - c) @ inl.T
            half = (np.percentile(pr, 98, 0) - np.percentile(pr, 2, 0)) / 2
            panel = dict(center=c.tolist(), u=inl[0].tolist(), v=inl[1].tolist(),
                         half=half.tolist())
        return dict(constraints=self.items, cons_path=str(self.cons_path),
                    up=up.tolist(), up_src=up_src, red_n=red_n.tolist(), panel=panel,
                    presets=presets, pivot=piv.tolist(), radius=rad, down=down, q_rng=q_rng,
                    s_min=float(s_all.min()),
                    s_max=float(s_all.max()), s_med=float(np.median(s_all)),
                    regions=len(self.regions), step=self.step,
                    regions_path=str(self.regions_path), zoned=self.zoned,
                    ids_path=str(self.ids_path))


PAGE = r"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><title>Nettoyage 3D</title>
<style>
 :root{--bg:#111418;--fg:#e8eaed;--mut:#9aa0a6;--acc:#4caf50;--brd:#2a2f36}
 body{margin:0;background:var(--bg);color:var(--fg);font:13px system-ui,sans-serif;overflow:hidden}
 #wrap{position:fixed;inset:0}
 #im{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;background:#000}
 #ov{position:absolute;inset:0;width:100%;height:100%;cursor:crosshair}
 #help{font-size:11px;color:var(--mut);margin:4px 0 0;white-space:pre-wrap}
 #camid{position:absolute;right:12px;bottom:10px;color:var(--mut);font-size:12px}
 #panel{position:absolute;left:10px;top:10px;background:rgba(17,20,24,.88);border:1px solid var(--brd);
   border-radius:8px;padding:10px 12px;width:320px;max-height:calc(100vh - 40px);overflow:auto}
 #panel h1{font-size:14px;margin:0 0 6px}
 .row{display:flex;align-items:center;gap:8px;margin:6px 0}
 input[type=range]{flex:1}
 button{background:#1d2229;color:var(--fg);border:1px solid var(--brd);border-radius:5px;padding:4px 8px;cursor:pointer}
 button.on{background:var(--acc);color:#000;border-color:var(--acc)}
 button:disabled{opacity:.4;cursor:default}
 #bc:not(:disabled){background:var(--acc);color:#000;border-color:var(--acc)}
 #info{color:var(--mut);font-size:12px;line-height:1.45;margin-top:6px}
 #msg{position:absolute;left:50%;bottom:16px;transform:translateX(-50%);background:rgba(17,20,24,.9);
   border:1px solid var(--brd);border-radius:6px;padding:6px 12px;display:none}
</style></head><body>
<div id="wrap"><img id="im" alt=""><canvas id="ov"></canvas></div>
<div id="panel">
 <h1>Nettoyage 3D : zones d'elagage</h1>
 <div class="row">s <input id="s" type="range" step="0.5"><span id="sv"></span></div>
 <div class="row">q0 <input id="q0" type="range" step="0.02"><span id="q0v"></span></div>
 <div class="row">q1 <input id="q1" type="range" step="0.02"><span id="q1v"></span></div>
 <div class="row"><button id="bq">q median</button><span style="color:var(--mut);font-size:11px">q en ecarts types (plage 0.5 a 99.5 % des donnees)</span></div>
 <div class="row">focale <input id="f" type="range" min="0.4" max="2.5" step="0.05" value="0.9"><span id="fv"></span></div>
 <div class="row"><button id="bc" disabled>Enregistrer la zone (Entree)</button><button id="bx" disabled>Effacer le trace (Echap)</button></div>
 <div class="row" id="pickrow" style="display:none"><button id="bm">Mode : lasso (P)</button><span id="hv" style="color:var(--mut);font-size:11px"></span></div>
 <div class="row"><button id="bu">Annuler la derniere action (U)</button><button id="br">Recharger (R)</button></div>
 <div id="info"></div>
 <h1 style="margin-top:10px">Contraintes</h1>
 <div class="row"><label><input type="checkbox" id="sh_vgrid"> grille verticale</label></div>
 <div class="row"><span style="width:58px">incl. 1</span><input type="range" id="vt1" min="-20" max="20" step="0.1" value="0"><b id="vt1v" style="width:46px;text-align:right">0.0</b></div>
 <div class="row"><span style="width:58px">incl. 2</span><input type="range" id="vt2" min="-20" max="20" step="0.1" value="0"><b id="vt2v" style="width:46px;text-align:right">0.0</b></div>
 <div class="row"><button id="vsave">Enregistrer la verticale</button></div>
 <div id="vinfo" style="font-size:11px;color:var(--mut)"></div>
 <div class="row"><label><input type="checkbox" id="sh_plane"> plan rouge</label>
  <label><input type="checkbox" id="sh_excl"> exclusions</label>
  <label><input type="checkbox" id="sh_walls"> murs</label></div>
 <div class="row"><button id="addbox">+ pave d exclusion</button><button id="addwall">+ mur</button></div>
 <div class="row" id="clist" style="flex-wrap:wrap"></div>
 <div id="cedit"></div>
 <details><summary>Commandes (celles de demo/3d.html)</summary><pre id="help">deplacement (zqsd, touches physiques)
- z/s : avancer / reculer
- q/d : translation laterale
- espace / ctrl : monter / descendre
- maj : plus vite
camera (fleches)
- gauche/droite : pivoter
- haut/bas : incliner
- a/e : roulis
- i/k et j/l : orbiter
souris
- clic droit + glisser : tourner autour du centre
- clic gauche + glisser : LASSO (courbe fermee)
- clic milieu + glisser : zoom
- molette : zoom vers le centre
- maj + molette : pan
lasso
- Entree : valider (apercu en vert)
- Echap : annuler le trace
- U : annuler la derniere zone
autre
- 0-9 : vues pre-chargees, -/+ : cycler
- V : vue courante dans l'URL
- R : recharger le checkpoint</pre></details>
</div>
<div id="msg"></div><div id="camid"></div>
<script>
const M = __META__;
const im = document.getElementById('im'), ov = document.getElementById('ov'), g = ov.getContext('2d');
const S = document.getElementById('s'), Fs = document.getElementById('f');
S.min = M.s_min; S.max = M.s_max; S.value = M.s_med;
const Q = [document.getElementById('q0'), document.getElementById('q1')];
Q.forEach((el,k)=>{ el.min=M.q_rng[k][0]; el.max=M.q_rng[k][1]; el.value=M.q_rng[k][2]; el.oninput=()=>{dirty=true; info();}; });
document.getElementById('bq').onclick=()=>{ Q.forEach((el,k)=>el.value=M.q_rng[k][2]); dirty=true; info(); };
function qv(){ return Q.map(el=>parseFloat(el.value)); }

// ── camera : axes (x droite, y bas, z devant) exprimes dans le monde + centre optique ──
// meme convention que demo/3d.html (OpenCV : y descend a l'ecran, z vers l'avant)
const sub3=(a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]], add3=(a,b)=>[a[0]+b[0],a[1]+b[1],a[2]+b[2]];
const mul3=(a,s)=>[a[0]*s,a[1]*s,a[2]*s], dot3=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const cross3=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const len3=a=>Math.hypot(a[0],a[1],a[2]), norm3=a=>{const n=len3(a)||1;return [a[0]/n,a[1]/n,a[2]/n];};
const rot3=(v,k,a)=>{const c=Math.cos(a),s=Math.sin(a),kv=cross3(k,v),d=dot3(k,v)*(1-c);
  return [v[0]*c+kv[0]*s+k[0]*d, v[1]*c+kv[1]*s+k[1]*d, v[2]*c+kv[2]*s+k[2]*d];};
let worldUp = norm3(M.up.slice());
const pivot = M.pivot.slice(), minRadius = 0.05*M.radius, maxRadius = 30*M.radius;
let ax = null, C = null, orbitRadius = 1;          // ax = [x, y, z] de la camera
let dirty = true, busy = false, last = {}, W = 0, H = 0;

function setE(E){ ax=[E[0].slice(0,3),E[1].slice(0,3),E[2].slice(0,3)];
  const t=[E[0][3],E[1][3],E[2][3]]; C=[0,1,2].map(j=>-(ax[0][j]*t[0]+ax[1][j]*t[1]+ax[2][j]*t[2]));
  orbitRadius=len3(sub3(C,pivot)); dirty=true; }
function viewE(){ const t=ax.map(r=>-dot3(r,C));
  return [ax[0][0],ax[0][1],ax[0][2],t[0], ax[1][0],ax[1][1],ax[1][2],t[1], ax[2][0],ax[2][1],ax[2][2],t[2], 0,0,0,1]; }
function lookAtPivot(pos){ // vise le pivot, haut = worldUp (repli sur le haut courant aux poles)
  const f=norm3(sub3(pivot,pos)); const up=Math.abs(dot3(f,worldUp))>0.9995 ? mul3(ax[1],-1) : worldUp;
  const r=norm3(cross3(f,up)); const d=cross3(f,r); ax=[r,d,f]; C=pos; dirty=true; }
function setCameraPreset(i){ // comme demo/3d.html : direction du preset, distance courante
  const E=M.presets[i].E, dir=E[2].slice(0,3);
  lookAtPivot(sub3(pivot, mul3(dir, orbitRadius)));
  document.getElementById('camid').textContent='cam '+i+'  '+M.presets[i].name; }
function orbitAroundPivot(dx,dy){ // rotation RIGIDE autour du pivot, garde aux poles
  const right=norm3(rot3(ax[0],worldUp,dx)); let off=rot3(sub3(C,pivot),worldUp,dx);
  const tilted=rot3(off,right,dy); let useDy=dy;
  if (Math.abs(dot3(norm3(tilted),worldUp))<0.995) off=tilted; else useDy=0;
  const spin=v=>useDy? rot3(rot3(v,worldUp,dx),right,useDy) : rot3(v,worldUp,dx);
  ax=ax.map(spin); C=add3(pivot,off); orbitRadius=len3(off); dirty=true; }
function zoomBy(f){ const off=sub3(C,pivot), r=len3(off)||1e-6;
  const k=Math.max(minRadius,Math.min(maxRadius,r*f))/r; C=add3(pivot,mul3(off,k)); orbitRadius=r*k; dirty=true; }
function panBy(tx,ty){ const m=add3(mul3(ax[0],tx),mul3(ax[1],ty));
  for(let i=0;i<3;i++) pivot[i]+=m[i]; C=add3(C,m); dirty=true; }
function translateLocal(tx,ty,tz){ C=add3(C,add3(add3(mul3(ax[0],tx),mul3(ax[1],ty)),mul3(ax[2],tz))); dirty=true; }
function rotateLocal(i,a){ const k=ax[i].slice(); ax=ax.map(v=>rot3(v,k,a)); dirty=true; } // = rotate4(inv, a, axe i)
function yawWorld(a){ ax=ax.map(v=>rot3(v,worldUp,a)); dirty=true; }

// ── rendu cote serveur ─────────────────────────────────────────────────────────
function resize(){ W=Math.floor(innerWidth/2)*2; H=Math.floor(innerHeight/2)*2; ov.width=W; ov.height=H; dirty=true; draw(); }
function camObj(){ return {E: viewE(), f: parseFloat(Fs.value)*W, W, H}; }
async function render(){
  if (!dirty || busy) return; dirty=false; busy=true;
  try{
    const r=await fetch('/render',{method:'POST',body:JSON.stringify({cam:camObj(),s:parseFloat(S.value),q:qv(),preview,show:SHOW,hover:pickMode?hover:null})});
    const b=await r.blob(), u=URL.createObjectURL(b); im.onload=()=>URL.revokeObjectURL(u); im.src=u;
    last=JSON.parse(r.headers.get('X-Info')); info();
  }catch(e){ msg('erreur de rendu : '+e); }
  busy=false;
}
function info(){
  document.getElementById('bc').disabled = !preview;
  document.getElementById('bx').disabled = !preview;
  document.getElementById('sv').textContent=parseFloat(S.value).toFixed(1);
  document.getElementById('fv').textContent=parseFloat(Fs.value).toFixed(2);
  document.getElementById('q0v').textContent=(+Q[0].value).toFixed(2);
  document.getElementById('q1v').textContent=(+Q[1].value).toFixed(2);
  document.getElementById('info').innerHTML=
    'checkpoint pas '+last.step+', '+last.total+' gaussiennes<br>'+
    last.regions+' zone(s), '+last.items+' contrainte(s), '+last.hidden+' gaussiennes exclues a cet etat'+(SHOW.excl?' (en rouge)':' (masquees)')+'<br>'+
    last.near_wall+' gaussiennes a portee des murs (en cyan)<br>'+
    (preview?'<b style="color:#4caf50">apercu : '+last.preview+' gaussiennes (Entree valide, Echap annule)</b><br>':'')+
    (M.zoned?last.n_ids+' gaussienne(s) supprimee(s) au clic<br>':'')+
    'rendu '+(last.ms||0).toFixed(0)+' ms<br><span style="font-size:11px">'+M.regions_path+'</span>';
  document.getElementById('hv').textContent = pickMode ? (last.hover>=0 ? 'sous le curseur : gaussienne '+last.hover+' (clic = supprimer)' : 'rien sous le curseur') : '';
}
function msg(t){const m=document.getElementById('msg'); m.textContent=t; m.style.display='block'; clearTimeout(msg.h); msg.h=setTimeout(()=>m.style.display='none',2500);}

// ── lasso (clic gauche) ────────────────────────────────────────────────────────
let pts=[], preview=null;
function draw(){ g.clearRect(0,0,W,H); if(typeof drawItems==='function') drawItems(); const P=preview||pts; if(P.length<2) return;
  g.strokeStyle='#4caf50'; g.lineWidth=2; g.setLineDash(preview?[]:[6,4]);
  g.beginPath(); g.moveTo(P[0][0],P[0][1]); for(const p of P) g.lineTo(p[0],p[1]); if(preview) g.closePath(); g.stroke(); }
// image, canevas et rendu ont la MEME taille en pixels : un point du lasso est un pixel de la camera envoyee
function xy(e){const r=ov.getBoundingClientRect(); return [(e.clientX-r.left)*W/r.width,(e.clientY-r.top)*H/r.height];}
function simplify(P){ const out=[P[0]]; for(const p of P){const q=out[out.length-1]; if(Math.hypot(p[0]-q[0],p[1]-q[1])>3) out.push(p);} return out; }
async function commit(){ if(!preview) return;
  const r=await fetch('/commit',{method:'POST',body:JSON.stringify({cam:camObj(),poly:preview,s:parseFloat(S.value),q:qv()})});
  const j=await r.json(); msg('zone ajoutee : '+j.n+' gaussiennes masquees au total a cet etat. Appliquee au prochain nettoyage de l entrainement.');
  preview=null; draw(); dirty=true; info(); }
async function undo(){ const r=await fetch('/undo',{method:'POST'}); const j=await r.json(); msg(j.msg||'rien a annuler'); dirty=true; }
// ── mode clic : une gaussienne a la fois ──
let pickMode=false, hover=null;
if(M.zoned) document.getElementById('pickrow').style.display='flex';
function setPick(on){ pickMode=on&&M.zoned; document.getElementById('bm').textContent='Mode : '+(pickMode?'clic':'lasso')+' (P)';
  document.getElementById('bm').classList.toggle('on',pickMode); ov.style.cursor=pickMode?'cell':'crosshair'; hover=null; clearTrace(); }
document.getElementById('bm').onclick=()=>setPick(!pickMode);
async function pickAt(p){ const r=await fetch('/pick',{method:'POST',body:JSON.stringify({cam:camObj(),s:parseFloat(S.value),q:qv(),x:p[0],y:p[1]})});
  const j=await r.json(); msg(j.i>=0?'gaussienne '+j.i+' supprimee (U pour la remettre)':'rien sous le curseur'); dirty=true; }
async function reload(){ msg('rechargement...'); await fetch('/reload',{method:'POST'}); msg('checkpoint recharge'); dirty=true; }
document.getElementById('bu').onclick=undo; document.getElementById('br').onclick=reload;
function clearTrace(){ preview=null; pts=[]; draw(); dirty=true; info(); }
document.getElementById('bc').onclick=commit; document.getElementById('bx').onclick=clearTrace;

// ── souris (reprise de demo/3d.html ; le clic gauche trace le lasso au lieu de l'effort) ──
let down=false, startX=0, startY=0;
ov.addEventListener('contextmenu',e=>e.preventDefault());
ov.addEventListener('mousedown',e=>{ e.preventDefault(); startX=e.clientX; startY=e.clientY;
  if(e.button===1) down=3; else if(e.button===2) down=4;
  else if(e.button===0 && pickMode){ pickAt(xy(e)); }
  else if(e.button===0){ down=5; preview=null; pts=[xy(e)]; draw(); dirty=true; } });
ov.addEventListener('mousemove',e=>{ e.preventDefault();
  if(down==4){ const dx=3.0*(e.clientX-startX)/innerWidth, dy=3.0*(e.clientY-startY)/innerHeight;
    orbitAroundPivot(-dx,-dy); startX=e.clientX; startY=e.clientY; }
  else if(down==3){ zoomBy(Math.exp((e.clientY-startY)*0.005)); startX=e.clientX; startY=e.clientY; }
  else if(down==5){ pts.push(xy(e)); draw(); }
  if(pickMode){ hover=xy(e); dirty=true; } });
addEventListener('mouseup',()=>{ if(down==5){ if(pts.length>=3){ preview=simplify(pts); dirty=true; } pts=[]; draw(); } down=false; });
addEventListener('wheel',e=>{ if(e.target!==ov) return; e.preventDefault();
  const scale=e.deltaMode==1?10:e.deltaMode==2?innerHeight:1;
  if(e.shiftKey){ const ps=orbitRadius*0.0002; panBy(e.deltaX*scale*ps, e.deltaY*scale*ps); }
  else { const sp=(e.ctrlKey||e.metaKey)?0.01:0.0015; zoomBy(Math.exp(e.deltaY*scale*sp)); } },{passive:false});

// ── clavier (reprise de demo/3d.html : codes PHYSIQUES, donc zqsd sur AZERTY) ───────
let activeKeys=[], currentCameraIndex=1;
document.querySelectorAll('input[type=range],button').forEach(el=>{
  el.addEventListener('change',()=>el.blur()); el.addEventListener('mouseup',()=>el.blur()); });
addEventListener('keydown',e=>{
  if(e.target.tagName==='INPUT' && e.target.type!=='range' && e.target.type!=='checkbox') return;
  if(e.target.tagName==='INPUT') e.target.blur();
  if((e.ctrlKey||e.metaKey)&&(e.code==='KeyS'||e.code==='KeyD')) e.preventDefault();
  if(e.code==='Space') e.preventDefault();
  if(!activeKeys.includes(e.code)) activeKeys.push(e.code);
  if(/^\d$/.test(e.key)){ currentCameraIndex=parseInt(e.key)%M.presets.length; setCameraPreset(currentCameraIndex); }
  if(['-','_'].includes(e.key)){ currentCameraIndex=(currentCameraIndex+M.presets.length-1)%M.presets.length; setCameraPreset(currentCameraIndex); }
  if(['+','='].includes(e.key)){ currentCameraIndex=(currentCameraIndex+1)%M.presets.length; setCameraPreset(currentCameraIndex); }
  if(e.code==='KeyV'){ location.hash='#'+JSON.stringify(viewE().map(k=>Math.round(k*1000)/1000)); msg('vue enregistree dans l URL'); }
  if(e.key==='Enter') commit();
  if(e.key==='Escape') clearTrace();
  if(e.code==='KeyU') undo();
  if(e.code==='KeyP') setPick(!pickMode);
  if(e.code==='KeyR' && !e.ctrlKey) reload();
});
addEventListener('keyup',e=>{ activeKeys=activeKeys.filter(k=>k!==e.code); });
addEventListener('blur',()=>{ activeKeys=[]; });

function keys(){
  const has=k=>activeKeys.includes(k);
  const shift=has('ShiftLeft')||has('ShiftRight'), ctrl=has('ControlLeft')||has('ControlRight');
  const sp=shift?0.1:0.02;
  if(has('ArrowLeft'))  yawWorld(0.01);
  if(has('ArrowRight')) yawWorld(-0.01);
  if(has('ArrowUp'))    rotateLocal(0, 0.005);
  if(has('ArrowDown'))  rotateLocal(0,-0.005);
  if(has('KeyQ')) rotateLocal(2, 0.01);          // touche a en AZERTY
  if(has('KeyE')) rotateLocal(2,-0.01);
  if(has('KeyW')) translateLocal(0,0, sp);       // z en AZERTY
  if(has('KeyS') && !ctrl) translateLocal(0,0,-sp);
  if(has('KeyA')) translateLocal(-sp,0,0);       // q en AZERTY
  if(has('KeyD') && !ctrl) translateLocal( sp,0,0);
  if(has('Space')) translateLocal(0,-sp,0);
  if(ctrl && !has('KeyS') && !has('KeyW')) translateLocal(0,sp,0);
  if(['KeyJ','KeyK','KeyL','KeyI'].some(has)){   // orbite autour d'un point a 4 unites devant
    const d=4, P=add3(C,mul3(ax[2],d));
    rotateLocal(1, has('KeyJ')?-0.05:has('KeyL')?0.05:0);
    rotateLocal(0, has('KeyI')?0.05:has('KeyK')?-0.05:0);
    C=sub3(P,mul3(ax[2],d));
  }
}

// ── contraintes : paves d'exclusion et murs (constraints3d.py) ─────────────────────────
// Axes de base : e1 = horizontale de la normale du panneau rouge (vers les cameras),
// e2 = up x e1, up = verticale. Seul le LACET (rotation autour de la verticale) est reglable :
// un mur est un rectangle VERTICAL (plan local (x, y) = (e2, up) tourne, normale horizontale),
// un pave a ses faces haute et basse HORIZONTALES. Les angles gardent trois composantes dans
// le fichier (tangage et roulis a zero) pour rester lisibles par constraints3d.py.
let items = M.constraints.slice(), sel = -1, nextId = 1 + Math.max(0, ...items.map(it=>it.id));
// contraintes deja enregistrees : tangage et roulis ramenes a zero (murs verticaux, paves a
// faces horizontales)
const SHOW = {plane:true, excl:false, walls:true, vgrid:false};
setTimeout(()=>{ let ch=false; for(const it of items){ if(it.angles[1]||it.angles[2]){ it.angles=[it.angles[0],0,0]; it.R=Rof(it); ch=true; } } if(ch) saveItems(); }, 0);
// base de la scene : up = verticale, e1 = horizontale de la normale du panneau rouge
function makeBase(up){ const n=M.red_n, e1=norm3(sub3(n,mul3(up,dot3(n,up)))); return {up, e1, e2:cross3(up,e1)}; }
let BASE = makeBase(worldUp);
function baseOf(type){ const B=BASE; return type==='wall' ? [B.e2,B.up,B.e1] : [B.e1,B.e2,B.up]; }
function Rof(it){ let a=baseOf(it.type).map(v=>v.slice()); const [yw,pt,rl]=it.angles.map(x=>x*Math.PI/180);
  a=a.map(v=>rot3(v,BASE.up,yw)); const k1=a[1].slice(); a=a.map(v=>rot3(v,k1,pt));
  const k0=a[0].slice(); a=a.map(v=>rot3(v,k0,rl));
  return [0,1,2].map(i=>[a[0][i],a[1][i],a[2][i]]); }       // COLONNES = axes locaux
function newItem(type){
  const ctr=add3(C,mul3(ax[2],Math.min(3,orbitRadius)));
  const it={id:nextId++, type, center:ctr, angles:[0,0,0],
    half: type==='wall'?[0.5,0.3,0.01]:[0.2,0.2,0.2], snap:0.08, spacing:0.04, thickness:0.015};
  it.R=Rof(it); items.push(it); sel=items.length-1; buildEditor(); saveItems(); }
let saveT=null;
function saveItems(){ clearTimeout(saveT); saveT=setTimeout(async()=>{
  await fetch('/constraints',{method:'POST',body:JSON.stringify({items})}); dirty=true; },250); dirty=true; }
function project(X){ const d=sub3(X,C), x=dot3(ax[0],d), y=dot3(ax[1],d), z=dot3(ax[2],d);
  if(z<0.05) return null; const f=parseFloat(Fs.value)*W; return [f*x/z+W/2, f*y/z+H/2]; }
function seg(A,B){ const p=project(A), q=project(B); if(!p||!q) return; g.beginPath(); g.moveTo(p[0],p[1]); g.lineTo(q[0],q[1]); g.stroke(); }
function corners(it){ const R=it.R, c=it.center, h=it.half, col=i=>[R[0][i],R[1][i],R[2][i]], out=[];
  for(const sx of [-1,1]) for(const sy of [-1,1]) for(const sz of (it.type==='wall'?[0]:[-1,1]))
    out.push(add3(add3(add3(c,mul3(col(0),sx*h[0])),mul3(col(1),sy*h[1])),mul3(col(2),sz*h[2])));
  return out; }
function drawVGrid(){ if(!SHOW.vgrid) return; const B=BASE, P=M.pivot;
  g.strokeStyle='rgba(255,235,59,0.85)'; g.lineWidth=1; g.setLineDash([]);
  for(const r of [0.8,1.6,2.4]) for(let k=0;k<16;k++){ const t=2*Math.PI*k/16;
    const b=add3(P,add3(mul3(B.e1,r*Math.cos(t)),mul3(B.e2,r*Math.sin(t)))); seg(add3(b,mul3(B.up,-2)),add3(b,mul3(B.up,2))); }
  for(const h of [-1.5,-0.5,0.5,1.5]) for(let k=0;k<48;k++){ const t0=2*Math.PI*k/48, t1=2*Math.PI*(k+1)/48, r=2.4;
    const p0=add3(add3(P,mul3(B.up,h)),add3(mul3(B.e1,r*Math.cos(t0)),mul3(B.e2,r*Math.sin(t0))));
    const p1=add3(add3(P,mul3(B.up,h)),add3(mul3(B.e1,r*Math.cos(t1)),mul3(B.e2,r*Math.sin(t1)))); seg(p0,p1); } }
function drawItems(){ drawVGrid();
  if(SHOW.plane && M.panel){ const P=M.panel; g.strokeStyle='#ff5252'; g.lineWidth=2; g.setLineDash([]);
    for(const [hu,hv,dash] of [[P.half[0],P.half[1],[]],[3*P.half[0],3*P.half[1],[6,6]]]){ g.setLineDash(dash);
      const cs=[[-1,-1],[1,-1],[1,1],[-1,1]].map(([a,b])=>add3(add3(P.center,mul3(P.u,a*hu)),mul3(P.v,b*hv)));
      for(let i=0;i<4;i++) seg(cs[i],cs[(i+1)%4]); } g.setLineDash([]); }
  items.forEach((it,i)=>{
    if(it.type==='box' && !SHOW.excl && i!==sel) return;
    if(it.type==='wall' && !SHOW.walls && i!==sel) return;
    g.strokeStyle = it.type==='box' ? '#ff9800' : '#26c6da'; g.lineWidth = i===sel?3:1.5;
    const k=corners(it);
    if(it.type==='box'){ const E=[[0,1],[2,3],[4,5],[6,7],[0,2],[1,3],[4,6],[5,7],[0,4],[1,5],[2,6],[3,7]];
      for(const [a,b] of E) seg(k[a],k[b]); }
    else { for(const [a,b] of [[0,1],[1,3],[3,2],[2,0]]) seg(k[a],k[b]);
      const R=it.R, n=[R[0][2],R[1][2],R[2][2]]; seg(it.center, add3(it.center,mul3(n,0.25)));
      g.setLineDash([3,4]); const s=it.snap; for(const sg of [-1,1]){ const kk=corners({...it,center:add3(it.center,mul3(n,sg*s)),half:[it.half[0]+s,it.half[1]+s,0]}); for(const [a,b] of [[0,1],[1,3],[3,2],[2,0]]) seg(kk[a],kk[b]); } g.setLineDash([]); }
  }); }

// ── editeur ────────────────────────────────────────────────────────────────────────
function slider(label, get, set, lo, hi, step){ const d=document.createElement('div'); d.className='row';
  d.innerHTML='<span style="width:58px">'+label+'</span><input type="range" min="'+lo+'" max="'+hi+'" step="'+step+'"><b style="width:46px;text-align:right"></b>';
  const el=d.querySelector('input'), vb=d.querySelector('b'); el.value=get(); vb.textContent=(+el.value).toFixed(2);
  el.addEventListener('input',()=>{ set(+el.value); vb.textContent=(+el.value).toFixed(2); const it=items[sel]; it.R=Rof(it); saveItems(); });
  el.addEventListener('change',()=>el.blur()); el.addEventListener('mouseup',()=>el.blur()); return d; }
function buildEditor(){
  const L=document.getElementById('clist'); L.innerHTML='';
  items.forEach((it,i)=>{ const b=document.createElement('button'); b.textContent=(it.type==='box'?'pave ':'mur ')+it.id;
    if(i===sel) b.classList.add('on'); b.onclick=()=>{ sel=(sel===i?-1:i); buildEditor(); dirty=true; }; L.appendChild(b); });
  const E=document.getElementById('cedit'); E.innerHTML=''; if(sel<0||!items[sel]) return;
  const it=items[sel], B=BASE, P=M.pivot;
  // position : decalage du centre depuis le pivot, le long de e1, e2, up
  const off=k=>dot3(sub3(it.center,P),[B.e1,B.e2,B.up][k]);
  const setOff=(k,v)=>{ const o=[0,1,2].map(j=>j===k?v:off(j)); it.center=add3(add3(add3(P,mul3(B.e1,o[0])),mul3(B.e2,o[1])),mul3(B.up,o[2])); };
  ['e1 (panneau)','e2','haut'].forEach((nm,k)=>E.appendChild(slider(nm,()=>off(k),v=>setOff(k,v),-8,8,0.01)));
  const hl = it.type==='wall' ? ['demi-larg.','demi-haut.'] : ['demi-x','demi-y','demi-z'];
  hl.forEach((nm,k)=>E.appendChild(slider(nm,()=>it.half[k],v=>{ it.half[k]=v; gridInfo(); },0.01,4,0.01)));
  // murs : normale horizontale ; paves : faces horizontales. Seul le lacet (rotation autour
  // de la verticale) reste libre, tangage et roulis sont nuls par construction.
  E.appendChild(slider('lacet',()=>it.angles[0],v=>it.angles[0]=v,-180,180,0.5));
  if(it.type==='wall'){ E.appendChild(slider('capture',()=>it.snap,v=>it.snap=v,0.005,0.5,0.005));
    E.appendChild(slider('pas grille',()=>it.spacing,v=>{ it.spacing=v; gridInfo(); },0.01,0.2,0.005));
    E.appendChild(slider('epaisseur',()=>(it.thickness??0.015),v=>it.thickness=v,0.002,0.08,0.001));
    const gi=document.createElement('div'); gi.id='gridinfo'; gi.style.cssText='font-size:11px;margin:2px 0 6px';
    E.appendChild(gi); gridInfo(); }
  const r=document.createElement('div'); r.className='row';
  r.innerHTML='<button id="cfront">Placer devant la vue</button><button id="cdel">Supprimer</button>';
  E.appendChild(r);
  document.getElementById('cfront').onclick=()=>{ it.center=add3(C,mul3(ax[2],Math.min(3,orbitRadius))); buildEditor(); saveItems(); };
  document.getElementById('cdel').onclick=()=>{ items.splice(sel,1); sel=-1; buildEditor(); saveItems(); };
}
// points de la grille du mur : au-dela de 50 000 (constraints3d.MAX_NEW), l'entrainement
// elargit le pas pour tenir sous ce plafond
function gridInfo(){ const el=document.getElementById('gridinfo'); const it=items[sel]; if(!el||!it||it.type!=='wall') return;
  const n=Math.ceil(2*it.half[0]/it.spacing)*Math.ceil(2*it.half[1]/it.spacing);
  el.innerHTML='grille : <b>'+n+'</b> points'+(n>50000?' <span style="color:#ff9800">au-dela de 50 000 : pas elargi a '+Math.sqrt(4*it.half[0]*it.half[1]/50000).toFixed(3)+' a l entrainement</span>':''); }
const UP0=worldUp.slice(), B0=makeBase(UP0); let tilt=[0,0];
function applyTilt(){ worldUp=norm3(rot3(rot3(UP0,B0.e1,tilt[0]*Math.PI/180),B0.e2,tilt[1]*Math.PI/180));
  BASE=makeBase(worldUp); items.forEach(it=>it.R=Rof(it)); dirty=true;
  document.getElementById('vinfo').textContent='source : '+M.up_src+(tilt[0]||tilt[1]?' + inclinaison '+tilt.map(x=>x.toFixed(1)).join(' / ')+' deg (non enregistree)':''); }
for(const [id,k] of [['vt1',0],['vt2',1]]){ const el=document.getElementById(id);
  el.addEventListener('input',()=>{ tilt[k]=+el.value; document.getElementById(id+'v').textContent=(+el.value).toFixed(1); applyTilt(); });
  el.addEventListener('change',()=>el.blur()); }
document.getElementById('vsave').onclick=async()=>{ await fetch('/vertical',{method:'POST',body:JSON.stringify({up:worldUp})});
  items.forEach(it=>it.R=Rof(it)); saveItems(); msg('verticale enregistree ; les murs et paves sont reorientes'); };
applyTilt();
document.getElementById('addbox').onclick=()=>newItem('box');
document.getElementById('addwall').onclick=()=>newItem('wall');
for(const k of ['plane','excl','walls','vgrid']){ const cb=document.getElementById('sh_'+k); cb.checked=SHOW[k];
  cb.onchange=()=>{ SHOW[k]=cb.checked; cb.blur(); dirty=true; }; }
buildEditor();

function frame(){ keys(); render(); draw(); requestAnimationFrame(frame); }

S.oninput=()=>{dirty=true; info();}; Fs.oninput=()=>{dirty=true; info();};
addEventListener('resize',resize);
// vue initiale : #hash (touche V) sinon l'iphone1 tel quel
let h=null; try{ h=JSON.parse(decodeURIComponent(location.hash.slice(1))); }catch(e){}
if(h && h.length===16) setE([h.slice(0,4),h.slice(4,8),h.slice(8,12),h.slice(12,16)]);
else setE(M.presets[1].E);
resize(); frame();
</script></body></html>
"""


def main():
    a = parse()
    T = Tool(a)
    page = PAGE.replace('__META__', json.dumps(T.meta()))

    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, body, ctype, extra=None):
            self.send_response(200)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _body(self):
            n = int(self.headers.get('Content-Length', 0))
            return json.loads(self.rfile.read(n) or b'{}')

        def do_GET(self):
            self._send(page.encode('utf-8'), 'text/html; charset=utf-8')

        def do_POST(self):
            try:
                if self.path == '/render':
                    b = self._body()
                    jpg, inf = T.render(b['cam'], float(b['s']), b.get('q'), b.get('preview'),
                                        b.get('show'), b.get('hover'))
                    self._send(jpg, 'image/jpeg', {'X-Info': json.dumps(inf)})
                elif self.path == '/commit':
                    b = self._body()
                    T.commit(b['cam'], b['poly'], b.get('s'), b.get('q'))
                    with T.lock:
                        n = int(T.state(float(b['s']), b.get('q'))['hidden'].sum())
                    self._send(json.dumps({'n': n}).encode(), 'application/json')
                elif self.path == '/vertical':
                    up = np.asarray(self._body()['up'], float)
                    up /= np.linalg.norm(up)
                    (HERE / a.vertical).write_text(json.dumps(
                        {'up': up.tolist(), 'note': 'reglee dans prune_tool.py, %s'
                         % time.strftime('%Y-%m-%d %H:%M:%S')}, indent=1), encoding='utf-8')
                    self._send(b'{}', 'application/json')
                elif self.path == '/constraints':
                    T.set_items(self._body()['items'])
                    self._send(b'{}', 'application/json')
                elif self.path == '/undo':
                    m = T.undo()
                    self._send(json.dumps({'msg': m}).encode(), 'application/json')
                elif self.path == '/pick':
                    b = self._body()
                    i = T.pick_commit(b['cam'], float(b['s']), b.get('q'), b['x'], b['y'])
                    self._send(json.dumps({'i': i}).encode(), 'application/json')
                elif self.path == '/reload':
                    T.load()
                    self._send(b'{}', 'application/json')
                else:
                    self.send_error(404)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send_error(500, str(e))

    srv = ThreadingHTTPServer(('127.0.0.1', a.port), H)
    url = 'http://127.0.0.1:%d/' % a.port
    print('outil : %s   (zones -> %s)' % (url, T.regions_path), flush=True)
    if not a.no_open:
        webbrowser.open(url)
    srv.serve_forever()


if __name__ == '__main__':
    main()
