"""
Demo 3D interactive du decodeur a deformation PARTAGEE (train_dec3d_shared.py) : on tourne
autour de la scene reconstruite, on tire sur la languette a la souris, et une physique
lineaire identifiee fait evoluer sa forme.

CHECKPOINT. Le dernier checkpoint de l'entrainement en cours est recharge automatiquement
(--reload-every, 30 s par defaut) : la demo s'affine au fil du run, sans relancer la page.

ETAT. z = (s, q0, q1). s est l'abscisse curviligne du bras, une ENTREE : le curseur s donne
une consigne que le bras suit avec une inertie (sinon un coup de curseur produirait une
acceleration infinie). q = (q0, q1) est la forme de la languette, regie par

    q̈ = -K q - C q̇ + G s̈ + B s + b + Jᵀ f

avec K, C, G, B, b ajustes par fit_dynamics_mpov.py (runs/dyn_lin/model.pt), meme recette que
la demo 2D d'img5175. L'integration se fait dans le navigateur ; le serveur ne fait que rendre.

EFFORT. Clic gauche sur la scene : le serveur selectionne les gaussiennes sous le curseur
(les plus proches de la camera, pour ne pas saisir le fond a travers l'objet), et renvoie le
point materiel saisi et J = d(pixel)/dq, par differences finies centrees sur q, dans la
camera COURANTE. En glissant, f = curseur - point saisi (en pixels), et F_q = Jᵀ f entre au
second membre. J est redemande en continu : il depend de l'etat et du point de vue.

GEL. Toucher un curseur q0 ou q1 gele la physique : q prend la valeur du curseur. Le bouton
Lecture la relance depuis cet etat. s reste pilotable pendant le gel.

BARRIERE. Comme la demo 2D : un potentiel nul dans le domaine des q mesures et un mur dur au
bord empechent q de partir la ou le decodeur n'a jamais rien vu.

CAMERA. Les commandes du viewer demo/3d.html du site (zqsd, fleches, a/e, ijkl, clic droit =
orbite autour du centre, molette = zoom, maj+molette = pan, 0-9 et -/+ = vues, V = vue dans
l'URL) ; le clic gauche applique l'effort, comme la-bas.

DECODEUR PAR ZONES (`train_dec3d_zoned.py`) : detecte dans le checkpoint et enveloppe par
`ZonedAdapter`, qui donne la meme interface `state(z, idx)` avec des couleurs en harmoniques
spheriques (le rendu passe alors `sh_degree` a gsplat). z(t) et l'encodeur sont ceux que le
run a utilises (`args['z_all']`, `runs/i1_d2_frozen/ae.pt` par defaut).

Usage :
    py demo3d.py --run dec3d_zoned                 # decodeur par zones
    py demo3d.py                                   # -> http://127.0.0.1:8783
    py demo3d.py --run dec3d_shared_p1 --phys runs/dyn_lin/model.pt --port 8783
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

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--run', default='dec3d_shared_p1')
    p.add_argument('--phys', default='runs/dyn_lin/model.pt')
    p.add_argument('--floor', default='runs/dec3d_shared/floor_plane.npz',
                   help='donne la verticale, pour une orbite qui ne roule pas')
    p.add_argument('--barrier-grid', type=int, default=192)
    p.add_argument('--barrier-close', type=int, default=5)
    p.add_argument('--reload-every', type=float, default=30.0,
                   help='secondes entre deux verifications du checkpoint ; un checkpoint '
                        'plus recent est recharge sans relancer la page (0 = jamais)')
    p.add_argument('--port', type=int, default=8783)
    p.add_argument('--near', type=float, default=0.2,
                   help="plan proche du rendu, celui de l'entrainement (0.2) : plus pres, des "
                        "gaussiennes jamais contraintes (a moins de 20 cm des cameras) deviennent "
                        "visibles")
    p.add_argument('--no-open', action='store_true')
    return p.parse_args()


def make_barrier(q, lo, hi, n, close_r):
    """Domaine couvert par les q mesures et distance a ce domaine, sur une grille n x n.
    Reprise telle quelle de img5175/serve_deform_demo.make_barrier."""
    ix = np.clip(((q - lo) / (hi - lo) * (n - 1)).round().astype(int), 0, n - 1)
    occ = np.zeros((n, n), np.uint8)
    occ[ix[:, 1], ix[:, 0]] = 1                       # ligne = q1, colonne = q0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_r + 1,) * 2)
    cov = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, k)
    nl, lab, st, _ = cv2.connectedComponentsWithStats(cov)
    if nl > 1:
        cov = (lab == (1 + int(np.argmax(st[1:, 4])))).astype(np.uint8)
    dpx = cv2.distanceTransform(1 - cov, cv2.DIST_L2, 5)
    cell = float(np.mean((hi - lo) / (n - 1)))
    dist = (dpx * cell).astype(np.float32)
    inside = float(np.mean([cov[j, i] for i, j in ix]))
    print('barriere : domaine couvert %.1f %% de la boite, %.2f %% des frames mesurees '
          'dehors' % (100 * cov.mean(), 100 * (1 - inside)))
    return dict(n=int(n), lo=lo.tolist(), hi=hi.tolist(),
                dist=np.round(dist, 4).ravel().tolist())


class ZonedAdapter:
    """Interface de SharedDeformSplat3D pour un ZonedSplat : state(z, idx) -> xyz, quat,
    demi-axes, opacite (K,), couleurs SH (B, K, (deg+1)^2, 3)."""

    def __init__(self, dec):
        self.dec = dec
        self.n_gaussians = int(dec.xyz.shape[0])
        self.sh_degree = int(dec.sh_degree)

    def state(self, z, idx=None):
        xyz, quat, logs, sh0 = self.dec.state(z)
        sh = torch.cat([sh0, self.dec.shN.unsqueeze(0).expand(z.shape[0], -1, -1, -1)], 2)
        alpha = torch.sigmoid(self.dec.logit_alpha)
        scale = torch.exp(logs)
        if idx is not None:
            return xyz[:, idx], quat[:, idx], scale[:, idx], alpha[idx], sh[:, idx]
        return xyz, quat, scale, alpha, sh


def load_decoder(ck):
    """(decodeur, sh_degree) depuis un checkpoint de train_dec3d_shared OU _zoned."""
    ar = ck['args']
    if 'idx_q' in ck['dec']:
        sys.path.insert(0, str(HERE))
        from train_dec3d_zoned import ZonedSplat, scene_from_ckpt
        sd = ck['dec']
        sc = scene_from_ckpt(sd, 3)          # taille du checkpoint (zone q doublee possible)
        dec = ZonedSplat(sc, sd['idx_s'], sd['idx_q'], sd['z_star'], ar['hidden'],
                         ar['layers'], ar['max_dxyz'], sd.get('carrier_coef'),
                         sd.get('carrier_rng'), sd.get('carrier_w')).cuda()
        # un checkpoint anterieur au porteur (ou a son raccord) n'a pas tous ses tampons :
        # ceux qui manquent gardent leur valeur par defaut (porteur nul, poids 1 sur la zone q)
        dec.load_state_dict(sd, strict=False)
        dec.eval()
        return ZonedAdapter(dec), int(dec.sh_degree)
    from models_deform import SharedDeformSplat3D
    dec = SharedDeformSplat3D(3, ck['dec']['xyz'].shape[0],
                              hidden=ar['hidden'], layers=ar['layers'],
                              pe_cond=ar['pe_cond'], alpha_min=ar['alpha_min'],
                              scale_min=ar['scale_min'], max_dxyz=ar['max_dxyz']).cuda()
    dec.load_state_dict(ck['dec'])
    dec.eval()
    return dec, None


class Demo:
    def __init__(self, a):
        from models_deform import SharedDeformSplat3D
        from scipy.signal import savgol_filter
        self.a = a
        self.lock = threading.Lock()
        self.run = HERE / 'runs' / a.run
        zf = self.run / 'z_all.npy'
        if not zf.exists():                  # run par zones : z(t) du run dont il reprend l etat
            zf = HERE / torch.load(self.run / 'dec.pt', map_location='cpu',
                                   weights_only=False)['args']['z_all']
        self.Z = np.load(zf).astype(np.float64)
        self.grabs, self.next_grab = {}, 1
        self.mtime, self.checked = 0.0, 0.0
        self.load()
        ar = self.args
        ae = torch.load(HERE / ar.get('init_ae', 'runs/i1_d2_frozen/ae.pt'), map_location='cpu',
                        weights_only=False)
        self.u_mean = float(np.atleast_1d(ae['u_mean'])[0])
        self.u_std = float(np.atleast_1d(ae['u_std'])[0])
        dec = self.dec

        m = torch.load(HERE / a.phys, map_location='cpu', weights_only=False)
        sd, pa, fps = m['model'], m['args'], float(m['fps'])
        s, q = self.Z[:, :1], self.Z[:, 1:]
        sdd = savgol_filter(s, pa['sg_win'] | 1, 3, deriv=2, delta=1.0 / fps, axis=0)
        K = sd['K'].numpy()
        f_hz = np.sqrt(np.abs(np.linalg.eigvals(K).real)) / (2 * np.pi)
        qlo, qhi = np.percentile(q, 1, 0), np.percentile(q, 99, 0)
        mid, half = 0.5 * (qlo + qhi), 0.5 * (qhi - qlo)
        rlo, rhi = mid - 1.4 * half, mid + 1.4 * half
        self.phys = dict(K=K.tolist(), C=sd['C'].numpy().tolist(),
                         G=sd['G'].numpy().tolist(), B=sd['B'].numpy().tolist(),
                         b=sd['b'].numpy().tolist(), dt=1.0 / fps,
                         udd_max=float(np.abs(sdd).max()), freq_hz=f_hz.tolist(),
                         q_std=q.std(0).tolist())
        self.barrier = make_barrier(q, rlo, rhi, a.barrier_grid, a.barrier_close)
        self.q_rng = [rlo.tolist(), rhi.tolist()]
        print('checkpoint pas %d, %d gaussiennes ; physique %s, frequences %s Hz'
              % (self.step, dec.n_gaussians, a.phys, np.round(f_hz, 2)), flush=True)

    # ── checkpoint : relu des que l'entrainement en ecrit un nouveau ────────────────
    def load(self):
        """Charge runs/<run>/dec.pt. L'entrainement l'ecrit pendant qu'on le lit parfois :
        on reessaie, et en cas d'echec on garde le decodeur courant."""
        from models_deform import SharedDeformSplat3D
        path = self.run / 'dec.pt'
        for k in range(5):
            try:
                mt = path.stat().st_mtime
                ck = torch.load(path, map_location='cuda', weights_only=False)
                break
            except Exception:
                time.sleep(2)
        else:
            return False
        ar = ck['args']
        dec, deg = load_decoder(ck)
        with self.lock:
            self.dec, self.step, self.args, self.mtime = dec, int(ck['step']), ar, mt
            self.sh_degree = deg
            # les saisies portent des indices de gaussiennes, qui changent avec l'elagage
            self.grabs = {}
        print('checkpoint pas %d, %d gaussiennes' % (self.step, dec.n_gaussians), flush=True)
        return True

    def maybe_reload(self):
        """Au plus une verification par --reload-every secondes (un stat du fichier)."""
        now = time.time()
        if self.a.reload_every <= 0 or now - self.checked < self.a.reload_every:
            return
        self.checked = now
        try:
            if (self.run / 'dec.pt').stat().st_mtime > self.mtime + 1:
                threading.Thread(target=self.load, daemon=True).start()
        except OSError:
            pass

    # ── camera ────────────────────────────────────────────────────────────────
    @staticmethod
    def camera(cam):
        W, H, f = int(cam['W']), int(cam['H']), float(cam['f'])
        E = torch.tensor(np.array(cam['E'], np.float32).reshape(4, 4), device='cuda')
        K = torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], dtype=torch.float32,
                         device='cuda')
        return E, K, W, H

    @staticmethod
    def project(xyz, E, K):
        Xc = xyz @ E[:3, :3].T + E[:3, 3]
        z = Xc[..., 2].clamp(min=1e-4)
        return torch.stack([K[0, 0] * Xc[..., 0] / z + K[0, 2],
                            K[1, 1] * Xc[..., 1] / z + K[1, 2]], -1), Xc[..., 2]

    def zt(self, z):
        return torch.tensor([z], dtype=torch.float32, device='cuda')

    # ── rendu ─────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def render(self, cam, z):
        from gsplat import rasterization
        E, K, W, H = self.camera(cam)
        t0 = time.time()
        with self.lock:
            xyz, quat, scale, alpha, color = self.dec.state(self.zt(z))
            r, _, _ = rasterization(means=xyz[0], quats=quat[0], scales=scale[0],
                                    opacities=alpha.clamp(0, 0.99), colors=color[0],
                                    viewmats=E[None], Ks=K[None], width=W, height=H,
                                    near_plane=self.a.near, far_plane=100.0,
                                    sh_degree=self.sh_degree, render_mode='RGB')
            img = (r[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        ok, buf = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, 88])
        return buf.tobytes(), (time.time() - t0) * 1000

    # ── saisie et effort ──────────────────────────────────────────────────────
    @torch.no_grad()
    def grab(self, cam, z, x, y, radius, uniform):
        """
        Gaussiennes saisies sous le curseur. Seules comptent les plus PROCHES de la camera
        (profondeur a moins de 15 % de la plus proche parmi les assez opaques) : en 3D le
        disque du curseur traverse toute la scene, et sans ce tri on tirerait aussi sur le
        mur du fond. Ponderation locale (empreinte x decroissance en rayon/2) ou globale
        (opacite seule), comme la demo 2D.

        Les statistiques de porte (voir jac) portent sur TOUTES les gaussiennes du disque,
        ponderees par leur seule opacite (`ws`), et non sur les poids d effort `w` : en
        saisie locale l empreinte concentre `w` sur 1 a 5 gaussiennes, meme sur l objet
        (mesure), donc n_eff y vaudrait toujours quelques unites.
        """
        E, K, W, H = self.camera(cam)
        with self.lock:
            xyz, _, scale, alpha, _ = self.dec.state(self.zt(z))
            uv, dep = self.project(xyz[0], E, K)
            d = uv - torch.tensor([x, y], device='cuda')
            r2 = (d * d).sum(1)
            m = (r2 <= radius * radius) & (dep > 0.05)
            if not bool(m.any()):
                return None
            opaque = m & (alpha > 0.2)
            zref = float(dep[opaque].min()) if bool(opaque.any()) else float(dep[m].min())
            m &= dep < zref * 1.15
            idx = torch.nonzero(m, as_tuple=True)[0]
            if uniform:
                w = alpha[idx].clone()
            else:
                s_px = scale[0][idx].mean(1) * K[0, 0] / dep[idx]
                foot = torch.exp(-0.5 * r2[idx] / s_px.clamp(min=0.5) ** 2)
                fall = torch.exp(-0.5 * r2[idx] / (radius * radius / 4.0))
                w = alpha[idx] * (foot + 1e-3) * fall
            keep = w > 1e-6
            if not bool(keep.any()):
                return None
            w = torch.where(keep, w, torch.zeros_like(w))
            w = w / w.sum()
            ws = alpha[idx].clone()
            ws = ws / ws.sum().clamp(min=1e-12)
        gid = self.next_grab
        self.next_grab += 1
        self.grabs[gid] = (idx, w, ws)
        for k in sorted(self.grabs)[:-16]:
            self.grabs.pop(k, None)
        pt, J, st = self.jac(cam, z, gid)
        return dict(id=gid, n=int(len(idx)), pt=pt, J=J, **st)

    @torch.no_grad()
    def jac(self, cam, z, gid, h=0.05):
        """
        Point saisi (pixels) et J = d(pixel)/d(q0, q1), 2 x 2, dans la camera courante.

        Statistiques de la selection, pour ecarter les artefacts (flotteurs isoles) :
          n_eff = 1 / sum w_i^2, nombre effectif de gaussiennes saisies ;
          coh   = |sum w_i J_i| / sum w_i |J_i| (normes de Frobenius), dans [0, 1] : 1 si
                  toutes les gaussiennes saisies bougent de la meme facon avec q, proche de
                  0 si leurs deplacements se compensent (bruit, flotteurs incoherents).
        En 3D la coherence est mesuree sur d(mu)/dq dans le repere de la piece, et non en
        pixels : elle ne depend donc pas du point de vue.
        """
        E, K, W, H = self.camera(cam)
        idx, w, ws = self.grabs[gid]
        with self.lock:
            base = self.project(self.dec.state(self.zt(z), idx)[0][0], E, K)[0]
            pt = (base * w[:, None]).sum(0)
            J = torch.zeros(2, 2, device='cuda')
            Jw = torch.zeros(len(idx), 3, 2, device='cuda')      # d(mu_i)/dq, monde
            for k in range(2):
                zp, zm = list(z), list(z)
                zp[1 + k] += h
                zm[1 + k] -= h
                xp = self.dec.state(self.zt(zp), idx)[0][0]
                xm = self.dec.state(self.zt(zm), idx)[0][0]
                Jw[:, :, k] = (xp - xm) / (2 * h)
                up, um = self.project(xp, E, K)[0], self.project(xm, E, K)[0]
                J[:, k] = (((up - um) / (2 * h)) * w[:, None]).sum(0)
            nrm = Jw.flatten(1).norm(dim=1)
            mean = (Jw * ws[:, None, None]).sum(0).norm()
            coh = float(mean / (nrm * ws).sum().clamp(min=1e-12))
            neff = float(1.0 / (ws * ws).sum())
        return pt.tolist(), J.tolist(), dict(neff=neff, coh=coh)

    def meta(self):
        rig = mv.load_rig('rig_manual.json')
        presets = []
        for c in ('iphone0', 'iphone1', 'iphone2'):
            E = np.eye(4)
            E[:3, :3] = np.array(rig['cameras'][c]['R'])
            E[:3, 3] = np.array(rig['cameras'][c]['t']).ravel()
            presets.append(dict(name=c, E=E.tolist()))
        i3 = json.loads((HERE / 'i3_poses_dense_s1.json').read_text(encoding='utf-8'))
        i3 = i3['iphone3_poses']
        keys = sorted(i3, key=float)
        for k in np.linspace(0, len(keys) - 1, 7).astype(int):
            pz = i3[keys[k]]
            E = np.eye(4)
            E[:3, :3] = np.array(pz['R'])
            E[:3, 3] = np.array(pz['t']).ravel()
            presets.append(dict(name='iphone3 t=%s' % keys[k], E=E.tolist()))
        with self.lock, torch.no_grad():
            xyz, _, _, alpha, _ = self.dec.state(self.zt(np.median(self.Z, 0).tolist()))
            X = xyz[0][alpha > 0.2].cpu().numpy()
        piv = np.median(X, 0)
        rad = float(np.percentile(np.linalg.norm(X - piv, axis=1), 90))
        vf = HERE / 'runs/dec3d_shared/vertical.json'     # reglee dans prune_tool.py
        if vf.exists():
            down = (-np.asarray(json.loads(vf.read_text(encoding='utf-8'))['up'])).tolist()
        elif (HERE / 'runs/dec3d_shared/up_from_surfels.npy').exists():
            down = (-np.load(HERE / 'runs/dec3d_shared/up_from_surfels.npy')).tolist()
        else:
            down = np.load(HERE / self.a.floor)['n'].tolist()
        s_raw = self.Z[:, 0] * self.u_std + self.u_mean
        # nuage des q mesures assimile a un disque : centre = moyenne, rayon = 99e centile des
        # distances au centre (le maximum serait tenu par une frame aberrante)
        q = self.Z[:, 1:]
        qc = q.mean(0)
        q_disk = dict(c=qc.tolist(),
                      r=float(np.percentile(np.linalg.norm(q - qc, axis=1), 99)))
        return dict(q_disk=q_disk, presets=presets, pivot=piv.tolist(), radius=rad, down=down,
                    phys=self.phys, barrier=self.barrier, q_rng=self.q_rng,
                    u_mean=self.u_mean, u_std=self.u_std,
                    s_lo=float(np.percentile(s_raw, 0.5)), s_hi=float(np.percentile(s_raw, 99.5)),
                    s_med=float(np.median(s_raw)), step=self.step,
                    gaussians=int(self.dec.n_gaussians))


PAGE = (HERE / 'demo3d_page.html')


def main():
    a = parse()
    D = Demo(a)
    meta = json.dumps(D.meta())

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
            # relue a chaque chargement : une modification de la page se voit sans relancer
            page = PAGE.read_text(encoding='utf-8').replace('__META__', meta)
            self._send(page.encode('utf-8'), 'text/html; charset=utf-8')

        def do_POST(self):
            try:
                b = self._body()
                if self.path == '/render':
                    D.maybe_reload()
                    jpg, ms = D.render(b['cam'], b['z'])
                    self._send(jpg, 'image/jpeg', {'X-Render-Ms': '%.1f' % ms,
                                                   'X-Step': str(D.step),
                                                   'X-Gaussians': str(D.dec.n_gaussians)})
                elif self.path == '/grab':
                    g = D.grab(b['cam'], b['z'], float(b['x']), float(b['y']),
                               float(b['r']), bool(b.get('uniform')))
                    self._send(json.dumps(g or {}).encode(), 'application/json')
                elif self.path == '/jac':
                    gid = int(b['id'])
                    if gid not in D.grabs:
                        self._send(b'{}', 'application/json')
                    else:
                        pt, J, st = D.jac(b['cam'], b['z'], gid)
                        self._send(json.dumps(dict(pt=pt, J=J, **st)).encode(),
                                   'application/json')
                else:
                    self.send_error(404)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send_error(500, str(e))

    srv = ThreadingHTTPServer(('127.0.0.1', a.port), H)
    url = 'http://127.0.0.1:%d/' % a.port
    print('demo : %s' % url, flush=True)
    if not a.no_open:
        webbrowser.open(url)
    srv.serve_forever()


if __name__ == '__main__':
    main()
