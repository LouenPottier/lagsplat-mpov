"""
3DGS STATIQUE de la piece, depuis la video de scene fixe de l'iphone3, dans le repere de
`rig_manual.json` (jauge |t01| = 1). Sert ensuite de FOND FIGE au decodeur a deformation
(`train_dec3d_shared.py`) : seules les gaussiennes de la zone du robot y seront apprises.

Entrees : `runs/static_i3/poses_room.json` et `points_room.npz` (`align_static_i3.py`),
images de la SfM (`runs/static_i3/sfm/images/`, 960 x 540).

Choix, et pourquoi :

  - MEME GEOMETRIE que `models_deform.SharedDeformSplat3D` : centre, quaternion (w, x, y,
    z), demi-axes, opacite. La couleur, elle, est en HARMONIQUES SPHERIQUES (`--sh-degree`,
    3 par defaut) : la piece a des reflets (ecran, sol, toles) qui dependent du point de
    vue ; sans SH l'optimisation les peint dans la couleur ou les imite par des flotteurs.
    Montee progressive du degre (+1 tous les `--sh-every` pas), comme dans 3DGS, pour que
    le degre 0 porte l'essentiel. Compatibilite avec le decodeur : une gaussienne RGB du
    decodeur s'ecrit en SH de degre 0 sans perte (dc = (rgb - 0.5) / C0, bandes superieures
    nulles), donc fond et robot se rasterisent toujours dans UN appel `gsplat`.
    ⚠️ Les SH d'ordre > 0 ne sont contraintes que dans les directions vues par l'iphone3 :
    a controler depuis les cameras fixes avant de figer ce fond.
  - IMAGES RECTIFIEES. Le rasteriseur est un stenope pur ; la distorsion SIMPLE_RADIAL de la
    SfM est retiree des images (meme K), les pixels sans source sont masques dans la perte.
  - CORRECTION DE COULEUR PAR IMAGE. L'iPhone regle son exposition en continu (luminance
    moyenne de 84 a 166 selon la frame) ; une affine par canal (gain, biais) par image,
    identite au depart et rappelee vers elle, absorbe ces ecarts au lieu de les peindre
    dans la scene.
  - RAFFINEMENT DES POSES (correction se(3) par image, petite, pas faible) : le recalage
    SfM -> piece est une similitude globale, mais la SfM elle-meme a son bruit de pose.
  - Densification et elagage standard (`gsplat.strategy.DefaultStrategy`).
  - HELD-OUT : une image sur `--val-every`. Ni pose ni couleur raffinees sur elles ; le PSNR
    held-out est donne apres un gain/biais par canal ajuste en forme close sur l'image
    (l'exposition d'une image jamais vue n'est pas previsible), et sans.

Sorties dans `runs/<out>/` : `scene.pt` (gaussiennes + poses raffinees + couleurs),
`scene.ply` (3DGS standard, lisible par les viewers usuels), `val_<pas>.png` (rendu contre
image held-out), `log.txt`.

Usage :
    py -u train_static3dgs.py --out static3dgs
"""
import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).parent.resolve()


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--poses', default='runs/static_i3/poses_room.json')
    p.add_argument('--points', default='runs/static_i3/points_room.npz')
    p.add_argument('--out', default='static3dgs')
    p.add_argument('--steps', type=int, default=30000)
    p.add_argument('--val-every', type=int, default=20)
    p.add_argument('--eval-every', type=int, default=2500)
    p.add_argument('--ssim-w', type=float, default=0.2)
    p.add_argument('--lr-means', type=float, default=1.6e-4,
                   help='multiplie par l etendue de la scene, comme dans 3DGS')
    p.add_argument('--lr-pose', type=float, default=1e-4)
    p.add_argument('--pose-start', type=int, default=1000)
    p.add_argument('--lr-color', type=float, default=1e-3)
    p.add_argument('--color-reg', type=float, default=1e-2)
    p.add_argument('--max-err', type=float, default=2.0, help='points SfM gardes a l init')
    p.add_argument('--refine-stop', type=int, default=15000)
    p.add_argument('--grow-grad2d', type=float, default=2e-4)
    p.add_argument('--exclude', nargs='*', default=[],
                   help='images ecartees, plages de frames "video:debut-fin" (video = base pour '
                        'la premiere prise, sinon le sous-dossier), ex. base:6185-6215. Sert aux '
                        'frames ou un objet tres proche masque la vue : vu sur trop peu d images, '
                        'il est reconstruit en nuage flottant devant les cameras voisines.')
    p.add_argument('--cam-clear', type=float, default=0.0,
                   help='rayon (jauge) autour de chaque centre de camera d entrainement ou toute '
                        'gaussienne est elaguee, tous les --cam-clear-every pas. 0 = desactive. '
                        'Contre les NAPPES collees a une camera (mesure sur le 1er run : 20 %% '
                        'des vues rendaient un ecran gris a 0.02-0.2 de la camera)')
    p.add_argument('--cam-clear-every', type=int, default=100)
    p.add_argument('--near', type=float, default=0.01,
                   help='plan proche du rasteriseur a l entrainement (jauge)')
    p.add_argument('--init-scene', default='',
                   help='part d une scene deja entrainee (ex. runs/static3dgs/scene_edit.pt, '
                        'nettoyee a la main dans static_points_tool.py) au lieu des seuls points '
                        'SfM. Les points SfM a plus de --seed-dist de toute gaussienne de cette '
                        'scene sont AJOUTES comme graines : c est ce qui ensemence les zones '
                        'nouvelles (plafond de la 2e video).')
    p.add_argument('--seed-dist', type=float, default=0.1)
    p.add_argument('--n-init', type=int, default=0,
                   help='nombre de gaussiennes au depart (sans --init-scene). Au-dela des '
                        'points SfM, des copies de points SfM tires au hasard, decalees a '
                        'l echelle de leurs voisins. 0 = les seuls points SfM.')
    p.add_argument('--max-add', type=int, default=-1,
                   help='BUDGET CUMULE de la densification : nombre maximal de gaussiennes '
                        'ajoutees sur tout le run (duplications + scissions), meme si des '
                        'elagages en retirent entre-temps. A chaque passage, seules les plus '
                        'forts gradients sont retenues dans la limite du reste. -1 = illimite.')
    p.add_argument('--init-poses', action='store_true',
                   help='avec --init-scene : reprend aussi les poses RAFFINEES et les '
                        'corrections de couleur de cette scene (images de meme nom), pour un '
                        'fine-tuning. Sans ca elles repartent de poses_room.json et de '
                        'l identite.')
    p.add_argument('--exclude-poses', default='runs/static_i3/excluded_poses.json',
                   help='images ecartees (poses absurdes, static_points_tool.py) ; vide = aucune')
    p.add_argument('--fixed-poses', default='',
                   help='poses corrigees (flip_poses_i3.py) : remplacent celles de --poses pour '
                        'les images listees (entrainement ET held-out), qui ne sont plus exclues')
    p.add_argument('--sh-degree', type=int, default=3)
    p.add_argument('--sh-every', type=int, default=1000)
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


# ───────────────────────────── utilitaires ─────────────────────────────

def ssim(x, y, win=11, sigma=1.5):
    """SSIM moyenne, x et y (B, 3, H, W) dans [0, 1]."""
    g = torch.exp(-(torch.arange(win, device=x.device) - win // 2) ** 2 / (2 * sigma ** 2))
    g = (g / g.sum()).float()
    k = (g[:, None] * g[None, :])[None, None].repeat(3, 1, 1, 1)
    mu_x = F.conv2d(x, k, groups=3)
    mu_y = F.conv2d(y, k, groups=3)
    sxx = F.conv2d(x * x, k, groups=3) - mu_x ** 2
    syy = F.conv2d(y * y, k, groups=3) - mu_y ** 2
    sxy = F.conv2d(x * y, k, groups=3) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))
    return m.mean()


def se3_exp(xi):
    """xi (N, 6) = (rotation vecteur, translation) -> (N, 4, 4)."""
    w, v = xi[:, :3], xi[:, 3:]
    th = w.norm(dim=1, keepdim=True).clamp(min=1e-9)
    k = w / th
    Kx = torch.zeros(len(xi), 3, 3, device=xi.device)
    Kx[:, 0, 1], Kx[:, 0, 2], Kx[:, 1, 2] = -k[:, 2], k[:, 1], -k[:, 0]
    Kx = Kx - Kx.transpose(1, 2)
    s, c = torch.sin(th)[..., None], torch.cos(th)[..., None]
    R = torch.eye(3, device=xi.device) + s * Kx + (1 - c) * Kx @ Kx
    T = torch.eye(4, device=xi.device).repeat(len(xi), 1, 1)
    T[:, :3, :3] = R
    T[:, :3, 3] = v
    return T


def knn_scale(xyz, k=4):
    """demi-axe initial : distance moyenne aux k plus proches voisins (par blocs)."""
    d_all = []
    for i in range(0, len(xyz), 4096):
        d = torch.cdist(xyz[i:i + 4096], xyz)
        d_all.append(d.topk(k + 1, largest=False).values[:, 1:].mean(1))
    return torch.cat(d_all)


C0 = 0.28209479177387814


def write_ply(path, xyz, quat, scale, alpha, f_dc, sh_rest):
    """PLY 3DGS standard (echelles et opacite en espace log / logit ; f_rest range par
    canal puis par coefficient, convention de l'implementation de reference)."""
    n = len(xyz)
    f_rest = sh_rest.transpose(0, 2, 1).reshape(n, -1)
    cols = [xyz, np.zeros((n, 3)), f_dc, f_rest,
            np.log(np.clip(alpha, 1e-6, 1 - 1e-6) / (1 - np.clip(alpha, 1e-6, 1 - 1e-6)))[:, None],
            np.log(scale), quat]
    data = np.concatenate(cols, 1).astype(np.float32)
    names = (['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2']
             + ['f_rest_%d' % k for k in range(f_rest.shape[1])] + ['opacity',
             'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3'])
    head = ('ply\nformat binary_little_endian 1.0\nelement vertex %d\n' % n
            + ''.join('property float %s\n' % nm for nm in names) + 'end_header\n')
    with open(path, 'wb') as fh:
        fh.write(head.encode('ascii'))
        fh.write(data.tobytes())


# ───────────────────────────── entrainement ─────────────────────────────

def main():
    a = parse()
    from gsplat import rasterization
    from gsplat.strategy import DefaultStrategy
    from gsplat.strategy.ops import remove
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = 'cuda'
    run = HERE / 'runs' / a.out
    run.mkdir(parents=True, exist_ok=True)
    logf = open(run / 'log.txt', 'a', encoding='utf-8')

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + '\n')
        logf.flush()

    P = json.loads((HERE / a.poses).read_text(encoding='utf-8'))
    Wd, Hd = P['size']
    K = np.array(P['K'], float)
    dist = np.array(P['dist'], float)
    def excluded(n):
        v = n.split('/')[0] if '/' in n else 'base'
        f = int(Path(n).stem[1:6])
        for e in a.exclude:
            ev, rg = e.split(':')
            lo, hi = (int(x) for x in rg.split('-'))
            if ev == v and lo <= f <= hi:
                return True
        return False

    bad = set()
    if a.exclude_poses and (HERE / a.exclude_poses).exists():
        bad = set(json.loads((HERE / a.exclude_poses).read_text(encoding='utf-8'))['names'])
    if a.fixed_poses:                  # une pose corrigee n'a plus de raison d'etre ecartee
        bad -= set(json.loads((HERE / a.fixed_poses).read_text(encoding='utf-8'))['poses'])
    names = sorted(n for n in P['poses'] if not excluded(n) and n not in bad)
    if bad:
        print('%d images ecartees par %s' % (len(bad), a.exclude_poses))
    if a.exclude:
        print('%d images ecartees par --exclude %s' % (len(P['poses']) - len(names), a.exclude))
    is_val = np.array([i % a.val_every == a.val_every // 2 for i in range(len(names))])

    # images rectifiees (meme K), masque des pixels valides
    mx, my = cv2.initUndistortRectifyMap(K, np.r_[dist, 0.0], None, K, (Wd, Hd), cv2.CV_32FC1)
    valid = cv2.remap(np.ones((Hd, Wd), np.float32), mx, my, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0.999
    imgs = np.zeros((len(names), Hd, Wd, 3), np.uint8)
    for i, nm in enumerate(names):
        im = cv2.imread(str(HERE / P['images_dir'] / nm))[:, :, ::-1]
        imgs[i] = cv2.remap(im, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    imgs_t = torch.from_numpy(imgs).to(dev)                    # (N, H, W, 3) uint8 sur GPU
    valid_t = torch.from_numpy(valid).to(dev).float()[None, ..., None]
    log('%d images %dx%d (%d held-out), rectifiees (k1 = %.4f), %.1f %% de pixels valides'
        % (len(names), Wd, Hd, is_val.sum(), dist[0], 100 * valid.mean()))

    E = np.stack([np.r_[np.c_[np.array(P['poses'][n]['R']), np.array(P['poses'][n]['t'])],
                        [[0, 0, 0, 1]]] for n in names])
    if a.fixed_poses:
        fp = json.loads((HERE / a.fixed_poses).read_text(encoding='utf-8'))['poses']
        n_fx = 0
        for i, n in enumerate(names):
            if n in fp:
                E[i] = np.array(fp[n]['viewmat'], float)
                n_fx += 1
        log('%d poses remplacees par %s (%d held-out)' % (
            n_fx, a.fixed_poses, int(sum(is_val[i] for i, n in enumerate(names) if n in fp))))
    E0 = torch.tensor(E, dtype=torch.float32, device=dev)
    Kt = torch.tensor(K, dtype=torch.float32, device=dev)
    centers = np.array([P['poses'][n]['center'] for n in names])
    scene_scale = float(np.linalg.norm(centers - centers.mean(0), axis=1).max() * 1.1)
    log('etendue des centres (x1.1) : %.3f unites de jauge' % scene_scale)

    pts = np.load(HERE / a.points)
    keep = pts['err'] < a.max_err
    xyz0 = torch.tensor(pts['xyz'][keep], dtype=torch.float32, device=dev)
    rgb0 = torch.tensor(pts['rgb'][keep] / 255.0, dtype=torch.float32, device=dev)
    log('init : %d points SfM (erreur < %.1f px) sur %d' % (keep.sum(), a.max_err, len(keep)))
    n_sh = (a.sh_degree + 1) ** 2 - 1
    if a.init_scene:
        sc = torch.load(HERE / a.init_scene, map_location=dev, weights_only=False)
        if int(sc['sh_degree']) != a.sh_degree:
            raise SystemExit('--init-scene en SH degre %d, --sh-degree %d'
                             % (sc['sh_degree'], a.sh_degree))
        from scipy.spatial import cKDTree
        d, _ = cKDTree(sc['xyz'].cpu().numpy()).query(xyz0.cpu().numpy())
        new = torch.tensor(d > a.seed_dist, device=dev)
        xs, rs = xyz0[new], rgb0[new]
        ss = knn_scale(xs).clamp(min=1e-4) if len(xs) > 5 else torch.full((len(xs),), 0.01,
                                                                           device=dev)
        log('init depuis %s : %d gaussiennes (pas %d) + %d graines SfM a plus de %.2f'
            % (a.init_scene, len(sc['xyz']), sc['step'], len(xs), a.seed_dist))
        cat = lambda u, v_: torch.cat([u, v_], 0)
        means = cat(sc['xyz'].float(), xs)
        scales = cat(torch.log(sc['scale'].float()), torch.log(ss)[:, None].repeat(1, 3))
        quats = cat(sc['quat'].float(), torch.tensor([1.0, 0, 0, 0], device=dev).repeat(len(xs), 1))
        opac = cat(torch.logit(sc['alpha'].float().clamp(1e-4, 1 - 1e-4)),
                   torch.logit(torch.full((len(xs),), 0.1, device=dev)))
        sh0 = cat(sc['sh'][:, :1].float(), ((rs - 0.5) / C0)[:, None, :])
        shN = cat(sc['sh'][:, 1:].float(), torch.zeros(len(xs), n_sh, 3, device=dev))
    else:
        s0 = knn_scale(xyz0).clamp(min=1e-4)
        if a.n_init > len(xyz0):
            g_ = torch.Generator(device=dev).manual_seed(a.seed)
            k_ = torch.randint(0, len(xyz0), (a.n_init - len(xyz0),), device=dev, generator=g_)
            jit = torch.randn(len(k_), 3, device=dev, generator=g_) * s0[k_, None]
            log('init completee a %d : %d copies decalees de points SfM'
                % (a.n_init, len(k_)))
            xyz0 = torch.cat([xyz0, xyz0[k_] + jit])
            rgb0 = torch.cat([rgb0, rgb0[k_]])
            s0 = torch.cat([s0, s0[k_]])
        means, scales = xyz0, torch.log(s0)[:, None].repeat(1, 3)
        quats = torch.tensor([1.0, 0, 0, 0], device=dev).repeat(len(xyz0), 1)
        opac = torch.logit(torch.full((len(xyz0),), 0.1, device=dev))
        sh0 = ((rgb0 - 0.5) / C0)[:, None, :]
        shN = torch.zeros(len(xyz0), n_sh, 3, device=dev)
    params = torch.nn.ParameterDict({
        'means': torch.nn.Parameter(means.contiguous()),
        'scales': torch.nn.Parameter(scales.contiguous()),
        'quats': torch.nn.Parameter(quats.contiguous()),
        'opacities': torch.nn.Parameter(opac.contiguous()),
        'sh0': torch.nn.Parameter(sh0.contiguous()),
        'shN': torch.nn.Parameter(shN.contiguous()),
    })
    lrs = dict(means=a.lr_means * scene_scale, scales=5e-3, quats=1e-3, opacities=5e-2,
               sh0=2.5e-3, shN=2.5e-3 / 20)
    optimizers = {k: torch.optim.Adam([{'params': params[k], 'lr': lrs[k], 'name': k}],
                                      eps=1e-15) for k in params}
    sched = torch.optim.lr_scheduler.ExponentialLR(optimizers['means'],
                                                   gamma=0.01 ** (1.0 / a.steps))
    class BudgetStrategy(DefaultStrategy):
        """DefaultStrategy dont la croissance est bornee par un budget CUMULE : a chaque
        passage, seules les gaussiennes au plus fort gradient sont dupliquees ou scindees,
        dans la limite du reste du budget (une scission ajoute une gaussienne nette, comme
        une duplication)."""
        budget = a.max_add
        added = 0

        @torch.no_grad()
        def _grow_gs(self, params, optimizers, state, step):
            if self.budget < 0:
                return super()._grow_gs(params, optimizers, state, step)
            left = self.budget - self.added
            grads = state['grad2d'] / state['count'].clamp_min(1)
            high = grads > self.grow_grad2d
            if left <= 0:
                return 0, 0
            if int(high.sum()) > left:
                keep = torch.zeros_like(high)
                keep[torch.topk(torch.where(high, grads, torch.zeros_like(grads)), left).indices] = True
                # le seuil gsplat est relu dans la classe mere : on masque les autres
                state['grad2d'] = torch.where(keep, state['grad2d'],
                                              torch.zeros_like(state['grad2d']))
            nd, ns = super()._grow_gs(params, optimizers, state, step)
            self.added += nd + ns
            return nd, ns

    strategy = BudgetStrategy(refine_stop_iter=a.refine_stop, grow_grad2d=a.grow_grad2d,
                              verbose=False)
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state(scene_scale=scene_scale)

    N = len(names)
    pose_d = torch.nn.Parameter(torch.zeros(N, 6, device=dev))
    col_A = torch.nn.Parameter(torch.ones(N, 3, device=dev))
    col_b = torch.nn.Parameter(torch.zeros(N, 3, device=dev))
    if a.init_poses and a.init_scene:
        sc_names = {n: i for i, n in enumerate(sc['names'])}
        n_p = 0
        with torch.no_grad():
            for i, n in enumerate(names):
                j = sc_names.get(n)
                if j is None:
                    continue
                if not is_val[i]:          # held-out : pose de poses_room.json, jamais raffinee
                    E0[i] = sc['viewmats'][j].to(dev).float()
                col_A[i] = sc['col_A'][j].to(dev)
                col_b[i] = sc['col_b'][j].to(dev)
                n_p += 1
        log('poses raffinees et couleurs reprises de %s pour %d / %d images'
            % (a.init_scene, n_p, N))
    opt_pose = torch.optim.Adam([pose_d], lr=a.lr_pose)
    opt_col = torch.optim.Adam([col_A, col_b], lr=a.lr_color)
    tr_idx = np.flatnonzero(~is_val)
    va_idx = np.flatnonzero(is_val)

    def viewmat(i, refine=True):
        if refine:
            return se3_exp(pose_d[i]) @ E0[i]
        return E0[i]

    def render(vm, Kb, deg=a.sh_degree):
        r, al, info = rasterization(
            means=params['means'], quats=params['quats'], scales=torch.exp(params['scales']),
            opacities=torch.sigmoid(params['opacities']),
            colors=torch.cat([params['sh0'], params['shN']], 1),
            viewmats=vm, Ks=Kb, width=Wd, height=Hd, near_plane=a.near, far_plane=100.0,
            sh_degree=deg, render_mode='RGB', packed=False)
        return r, al, info

    @torch.no_grad()
    def evaluate(step):
        ps_raw, ps_fit = [], []
        for i in va_idx:
            r, _, _ = render(viewmat(torch.tensor([i], device=dev), False), Kt[None])
            x = r[0].clamp(0, 1)
            y = imgs_t[i].float() / 255.0
            m = valid_t[0, ..., 0] > 0
            ps_raw.append(-10 * torch.log10(((x - y) ** 2)[m].mean()).item())
            # gain et biais par canal en forme close (exposition inconnue d une image jamais vue)
            xf, yf = x[m], y[m]
            xm, ym = xf.mean(0), yf.mean(0)
            g = ((xf - xm) * (yf - ym)).sum(0) / ((xf - xm) ** 2).sum(0).clamp(min=1e-9)
            xc = (g * (xf - xm) + ym).clamp(0, 1)
            ps_fit.append(-10 * torch.log10(((xc - yf) ** 2).mean()).item())
        # planche : 4 vues held-out, rendu / image
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        pick = va_idx[np.linspace(0, len(va_idx) - 1, 4).astype(int)]
        fig, ax = plt.subplots(2, 4, figsize=(20, 6))
        for j, i in enumerate(pick):
            r, _, _ = render(viewmat(torch.tensor([i], device=dev), False), Kt[None])
            ax[0, j].imshow(r[0].clamp(0, 1).cpu().numpy())
            ax[0, j].set_title('rendu, %s' % names[i], fontsize=9)
            ax[1, j].imshow(imgs[i])
            ax[1, j].set_title('image held-out', fontsize=9)
            for x in ax[:, j]:
                x.axis('off')
        fig.suptitle('pas %d : PSNR held-out %.2f dB brut, %.2f dB apres gain/biais par canal'
                     % (step, np.mean(ps_raw), np.mean(ps_fit)))
        fig.tight_layout()
        fig.savefig(run / ('val_%05d.png' % step), dpi=80)
        plt.close(fig)
        return float(np.mean(ps_raw)), float(np.mean(ps_fit))

    def save(step):
        with torch.no_grad():
            q = F.normalize(params['quats'], dim=-1)
            sc = torch.exp(params['scales'])
            al = torch.sigmoid(params['opacities'])
            rgb = (params['sh0'][:, 0] * C0 + 0.5).clamp(0, 1)
            vm = torch.stack([viewmat(torch.tensor([i], device=dev), not is_val[i])[0]
                              for i in range(N)])
        torch.save(dict(step=step, xyz=params['means'].detach().cpu(), quat=q.cpu(),
                        scale=sc.cpu(), alpha=al.cpu(), color=rgb.cpu(), sh_degree=a.sh_degree,
                        sh=torch.cat([params['sh0'], params['shN']], 1).detach().cpu(),
                        names=names, is_val=is_val, viewmats=vm.cpu(), K=Kt.cpu(),
                        size=(Wd, Hd), col_A=col_A.detach().cpu(), col_b=col_b.detach().cpu(),
                        scene_scale=scene_scale, args=vars(a),
                        note='repere de la piece (rig_manual.json, jauge |t01| = 1) ; quat '
                             '(w, x, y, z) ; scale = demi-axes ; sh (N, (deg+1)^2, 3) au '
                             'format gsplat ; color = partie DC seule, en RGB'), run / 'scene.pt')
        write_ply(run / 'scene.ply', params['means'].detach().cpu().numpy(),
                  q.cpu().numpy(), sc.cpu().numpy(), al.cpu().numpy(),
                  params['sh0'][:, 0].detach().cpu().numpy(),
                  params['shN'].detach().cpu().numpy())

    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    for step in range(a.steps):
        i = int(rng.choice(tr_idx))
        ii = torch.tensor([i], device=dev)
        vm = viewmat(ii, step >= a.pose_start)
        r, al, info = render(vm, Kt[None], min(a.sh_degree, step // a.sh_every))
        strategy.step_pre_backward(params, optimizers, state, step, info)
        out = r[0] * col_A[i] + col_b[i]
        tgt = imgs_t[i].float() / 255.0
        l1 = ((out - tgt).abs() * valid_t[0]).sum() / (valid_t.sum() * 3)
        ls = 1 - ssim((out * valid_t[0]).permute(2, 0, 1)[None],
                      (tgt * valid_t[0]).permute(2, 0, 1)[None])
        reg = ((col_A[i] - 1) ** 2).sum() + (col_b[i] ** 2).sum()
        loss = (1 - a.ssim_w) * l1 + a.ssim_w * ls + a.color_reg * reg
        loss.backward()
        for o in optimizers.values():
            o.step()
            o.zero_grad(set_to_none=True)
        opt_col.step()
        opt_col.zero_grad(set_to_none=True)
        if step >= a.pose_start:
            opt_pose.step()
        opt_pose.zero_grad(set_to_none=True)
        sched.step()
        strategy.step_post_backward(params, optimizers, state, step, info, packed=False)
        if a.cam_clear > 0 and (step + 1) % a.cam_clear_every == 0:
            with torch.no_grad():
                Cc = torch.linalg.inv(torch.stack([viewmat(torch.tensor([k], device=dev),
                                                           step >= a.pose_start)[0]
                                                   for k in tr_idx]))[:, :3, 3]
                near = torch.zeros(len(params['means']), dtype=torch.bool, device=dev)
                for i0 in range(0, len(Cc), 256):
                    near |= (torch.cdist(params['means'], Cc[i0:i0 + 256]).min(1).values
                             < a.cam_clear)
            n_clear = int(near.sum())
            if n_clear:
                remove(params, optimizers, state, near)
            if (step + 1) % 500 == 0:
                log('  elagage pres des cameras : %d gaussiennes' % n_clear)
        if step % 500 == 0:
            with torch.no_grad():
                dp = pose_d[tr_idx]
            log('pas %5d  L1 %.4f  SSIM %.3f  %7d gaussiennes (%d ajoutees)  pose |w| med %.3f '
                'deg |v| med %.4f  %.0f s'
                % (step, l1.item(), 1 - ls.item(), len(params['means']), strategy.added,
                   math.degrees(dp[:, :3].norm(dim=1).median().item()),
                   dp[:, 3:].norm(dim=1).median().item(), time.time() - t0))
        if (step + 1) % a.eval_every == 0 or step + 1 == a.steps:
            pr, pf = evaluate(step + 1)
            log('  held-out pas %d : PSNR %.2f dB brut, %.2f dB apres gain/biais'
                % (step + 1, pr, pf))
            save(step + 1)
    log('termine en %.0f s -> %s' % (time.time() - t0, run / 'scene.pt'))


main()
