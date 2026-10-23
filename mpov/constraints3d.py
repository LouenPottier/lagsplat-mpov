"""
Contraintes geometriques 3D du decodeur partage, dessinees dans prune_tool.py et appliquees par
train_dec3d_shared.py --constraints. Module partage par les deux, pour que l'outil montre
exactement ce que l'entrainement fera.

Fichier <run>/constraints.json : {"items": [...]}, chaque element

  {"id": int, "type": "box" | "wall", "center": [x, y, z], "half": [hx, hy, hz],
   "R": 3 x 3 (COLONNES = axes locaux exprimes dans le monde), "angles": [lacet, tangage,
   roulis] en degres (pour l'outil seulement), "snap": d, "spacing": p}

- PAVE ("box") : zone d'exclusion. Toute gaussienne dont le centre tombe dedans est supprimee.
- MUR ("wall") : rectangle dans le plan local (x, y), normale = axe local z, demi-cotes
  half[0], half[1] (half[2] ignore). Les gaussiennes a moins de `snap` du rectangle sont
  rattachees au mur : projetees sur le plan, aplaties, alignees sur lui, et privees de toute
  dependance a l'etat (leurs lignes de sortie du MLP sont mises a zero). Les parties du
  rectangle qu'aucune gaussienne ne couvre en recoivent de nouvelles, sur une grille de pas
  `spacing`. Leur POSITION est fixee (point de rattachement ou de creation) ; couleur,
  opacite et demi-axes dans le plan restent appris. Ce sont des gaussiennes 3D ordinaires, FINES : deux demi-axes dans le plan,
  demi-axe normal = `thickness` (defaut 0.015, unites de la jauge), et un decalage fixe le
  long de la normale, aleatoire dans +-10 % de l'epaisseur, pour que le tri en profondeur ne
  bascule pas d'une vue a l'autre (artefacts quand la camera bouge).

  Un mur n'ajoute jamais plus de MAX_NEW gaussiennes par nettoyage : si sa grille en demande
  davantage, le pas est ELARGI (et les gaussiennes agrandies d'autant) jusqu'a tenir sous
  MAX_NEW, et le journal le dit. Supprimer un mur supprime les
  gaussiennes qu'il a CREEES (`born`) et relache celles qu'il avait seulement rattachees.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def load_items(path):
    p = Path(path)
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding='utf-8')).get('items', [])


def save_items(path, items):
    Path(path).write_text(json.dumps({'items': items}, indent=1), encoding='utf-8')


def frame(it, dev='cpu'):
    c = torch.tensor(it['center'], dtype=torch.float32, device=dev)
    R = torch.tensor(it['R'], dtype=torch.float32, device=dev)
    h = torch.tensor(it['half'], dtype=torch.float32, device=dev)
    return c, R, h


def inside_box(xyz, it):
    """(K, 3) -> (K,) bool : centre dans le pave."""
    c, R, h = frame(it, xyz.device)
    loc = (xyz - c) @ R
    return (loc.abs() <= h).all(-1)


def inside_boxes(xyz, items):
    m = torch.zeros(xyz.shape[0], dtype=torch.bool, device=xyz.device)
    for it in items:
        if it['type'] == 'box':
            m |= inside_box(xyz, it)
    return m


def near_wall(xyz, it):
    """(K, 3) -> (K,) bool : a moins de `snap` du rectangle du mur."""
    c, R, h = frame(it, xyz.device)
    loc = (xyz - c) @ R
    s = float(it.get('snap', 0.05))
    return (loc[:, 2].abs() < s) & (loc[:, 0].abs() < h[0] + s) & (loc[:, 1].abs() < h[1] + s)


def rot_to_quat(R):
    """Matrice de rotation 3 x 3 (numpy) -> quaternion wxyz."""
    R = np.asarray(R, float)
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def softplus_inv(x):
    x = x.clamp(min=1e-6)
    return x + torch.log(-torch.expm1(-x).clamp(max=-1e-12))


CANON = ('xyz', 'logs_raw', 'quat_raw', 'log_alpha', 'color_raw')
MAX_NEW = 50000
THICKNESS = 0.015
# Decalage le long de la normale, fixe par gaussienne, tire dans +-JITTER x epaisseur. Toutes
# les gaussiennes d'un mur exactement dans le meme plan ont la meme profondeur a un point de
# vue donne : l'ordre du tri de gsplat y bascule d'une vue a l'autre et les disques se
# chevauchent en scintillant quand la camera bouge.
JITTER = 0.1


def thickness_raw(it, scale_min):
    """logs_raw du demi-axe normal : scale = scale_min + softplus(raw) = epaisseur."""
    t = max(float(it.get('thickness', THICKNESS)) - scale_min, 1e-5)
    return float(softplus_inv(torch.tensor(t)))


def resize(dec, opt, keep=None, add=None):
    """
    Garde les gaussiennes `keep` (indices) et ajoute celles de `add` (dict des parametres
    canoniques, meme cles que CANON, n lignes chacun). Les lignes de sortie du MLP des
    gaussiennes ajoutees sont a ZERO (aucune dependance a l'etat) ; l'etat d'Adam est coupe
    aux memes indices et mis a zero pour les nouvelles.

    ⚠️ NOUVEAUX objets Parameter, et non `p.data = ...` : autograd garde la forme d'un
    parametre dans son noeud d'accumulation, et le backward suivant echoue.
    """
    K, n_out = dec.n_gaussians, dec.N_OUT
    dev = dec.xyz.device
    if keep is None:
        keep = torch.arange(K, device=dev)
    n_add = 0 if add is None else len(add['xyz'])
    rows = (keep[:, None] * n_out + torch.arange(n_out, device=dev)).reshape(-1)
    specs = [(dec, k, keep, (add[k] if add else None)) for k in CANON]
    Hd = dec.field.head.weight.shape[1]
    specs += [(dec.field.head, 'weight', rows,
               torch.zeros(n_add * n_out, Hd, device=dev) if n_add else None),
              (dec.field.head, 'bias', rows,
               torch.zeros(n_add * n_out, device=dev) if n_add else None)]
    for mod, name, ix, extra in specs:
        old = getattr(mod, name)
        data = old.data[ix]
        if extra is not None and len(extra):
            data = torch.cat([data, extra.to(data.dtype)])
        new = torch.nn.Parameter(data.contiguous())
        setattr(mod, name, new)
        st = opt.state.pop(old, None)
        if st:
            for k in ('exp_avg', 'exp_avg_sq'):
                if k in st:
                    v = st[k][ix]
                    if extra is not None and len(extra):
                        v = torch.cat([v, torch.zeros((len(extra),) + v.shape[1:],
                                                      device=v.device, dtype=v.dtype)])
                    st[k] = v.contiguous()
            opt.state[new] = st
        for g in opt.param_groups:
            g['params'] = [new if q is old else q for q in g['params']]
    dec.n_gaussians = dec.field.n_gaussians = len(keep) + n_add


class Walls:
    """
    Gaussiennes rattachees aux murs : `wall_id` (K,) vaut l'id du mur ou -1. Tenu a jour a
    travers les elagages (`index`) et les ajouts (`append`) ; sauvegarde dans le checkpoint.
    """

    def __init__(self, dec, wall_id=None, born=None, anchor=None):
        K = dec.n_gaussians
        dev = dec.xyz.device

        def init(v):
            return (torch.full((K,), -1, dtype=torch.long, device=dev) if v is None
                    else torch.as_tensor(v, dtype=torch.long, device=dev))
        self.id, self.born = init(wall_id), init(born)
        assert len(self.id) == K and len(self.born) == K, (len(self.id), len(self.born), K)
        # POSITION FIXEE des gaussiennes de mur : le point ou elles ont ete rattachees ou
        # creees. Sans elle, elles glissaient dans le plan (0.19 pas de mediane en 2000 pas),
        # rouvraient des trous que chaque nettoyage recomblait, et le mur grossissait sans fin
        # (40 358 puis 45 823 gaussiennes). Checkpoint ancien sans ancre : position courante.
        self.anchor = (torch.as_tensor(anchor, dtype=torch.float32, device=dev).clone()
                       if anchor is not None else dec.xyz.data.clone())
        self.walls = []

    def orphans(self, items):
        """Gaussiennes CREEES par un mur qui n'existe plus : a supprimer."""
        ids = torch.tensor([int(it['id']) for it in items if it['type'] == 'wall'] or [-2],
                           device=self.born.device)
        return (self.born >= 0) & ~torch.isin(self.born, ids)

    def attached(self):
        return self.id >= 0

    def index(self, keep):
        self.id = self.id[keep]
        self.born = self.born[keep]
        self.anchor = self.anchor[keep]

    @torch.no_grad()
    def enforce(self, dec):
        """Apres chaque pas : sur le plan, dans le rectangle, alignees, sans deformation."""
        if not self.walls or not bool(self.attached().any()):
            return
        n_out = dec.N_OUT
        W = dec.field.head.weight.data.view(dec.n_gaussians, n_out, -1)
        b = dec.field.head.bias.data.view(dec.n_gaussians, n_out)
        for w in self.walls:
            m = self.id == int(w['id'])
            if not bool(m.any()):
                continue
            c, R, h = frame(w, dec.xyz.device)
            loc = (self.anchor[m] - c) @ R
            t = JITTER * float(w.get('thickness', THICKNESS))
            loc[:, 2] = loc[:, 2].clamp(-t, t)            # decalage normal propre a chacune
            loc[:, 0] = loc[:, 0].clamp(-h[0], h[0])
            loc[:, 1] = loc[:, 1].clamp(-h[1], h[1])
            dec.xyz.data[m] = c + loc @ R.T
            self.anchor[m] = dec.xyz.data[m]
            q = torch.tensor(rot_to_quat(w['R']), dtype=torch.float32, device=dec.xyz.device)
            dec.quat_raw.data[m] = q
            dec.logs_raw.data[m, 2] = thickness_raw(w, dec.scale_min)
            W[m] = 0
            b[m] = 0

    @torch.no_grad()
    def update(self, dec, opt, items, z_med, log=print):
        """
        A chaque nettoyage : libere les gaussiennes des murs supprimes, rattache les
        gaussiennes proches (a l'etat median : position, couleur et demi-axes pris A CET
        ETAT, pour que le rattachement ne change pas le rendu), puis comble les trous.
        """
        self.walls = [it for it in items if it['type'] == 'wall']
        ids = torch.tensor([int(w['id']) for w in self.walls] or [-2], device=self.id.device)
        gone = (self.id >= 0) & ~torch.isin(self.id, ids)
        if bool(gone.any()):
            self.id[gone] = -1
        if not self.walls:
            return
        dev = dec.xyz.device
        for w in self.walls:
            # etat recalcule a chaque mur : le mur precedent a pu AJOUTER des gaussiennes
            xyz_m, _, scale_m, alpha, color_m = dec.state(z_med)
            xyz_m, scale_m, color_m = xyz_m[0], scale_m[0], color_m[0]
            wid = int(w['id'])
            near = near_wall(xyz_m, w) & (self.id < 0)
            n_near = int(near.sum())
            if n_near:
                self.id[near] = wid
                dec.xyz.data[near] = xyz_m[near]
                c0, R0, _ = frame(w, dev)
                l0 = (xyz_m[near] - c0) @ R0
                l0[:, 2] = 0                               # sur le plan ; decale plus bas
                self.anchor[near] = c0 + l0 @ R0.T
                dec.color_raw.data[near] = torch.logit(color_m[near].clamp(1e-3, 1 - 1e-3))
                s2 = torch.sort(scale_m[near], -1).values[:, 1:].mean(-1, keepdim=True)
                dec.logs_raw.data[near, :2] = softplus_inv((s2 - dec.scale_min).clamp(min=1e-5))
            self.enforce(dec)
            # trous : points de grille sans gaussienne du mur a moins de 0.75 pas
            c, R, h = frame(w, dev)
            sp = float(w.get('spacing', 0.03))
            n_grid = np.ceil(2 * float(h[0]) / sp) * np.ceil(2 * float(h[1]) / sp)
            if n_grid > MAX_NEW:
                sp_old = sp
                sp = float(np.sqrt(4 * float(h[0]) * float(h[1]) / MAX_NEW)) * 1.02
                log('  mur %d : grille de %d points au pas %.3f, au-dela de %d : pas elargi a '
                    '%.3f' % (wid, n_grid, sp_old, MAX_NEW, sp))
            us = torch.arange(-float(h[0]) + sp / 2, float(h[0]), sp, device=dev)
            vs = torch.arange(-float(h[1]) + sp / 2, float(h[1]), sp, device=dev)
            G = torch.stack(torch.meshgrid(us, vs, indexing='ij'), -1).reshape(-1, 2)
            mine = self.id == wid
            covered = torch.zeros(len(G), dtype=torch.bool, device=dev)
            if bool(mine.any()):
                P = ((dec.xyz.data[mine] - c) @ R)[:, :2]
                for i in range(0, len(G), 4096):
                    covered[i:i + 4096] = torch.cdist(G[i:i + 4096], P).min(1).values < 0.75 * sp
            # ⚠️ jamais de point de grille dans un pave d'exclusion : les paves priment sur les
            # murs, donc une gaussienne creee la etait supprimee au nettoyage suivant, puis
            # recreee, etc. Mesure : ~16 500 gaussiennes neuves par nettoyage, qui n'avaient
            # jamais plus de 1000 pas pour apprendre (« une partie du mur ne bouge pas »).
            Xg = c + G[:, 0:1] * R[:, 0] + G[:, 1:2] * R[:, 1]
            in_box = inside_boxes(Xg, items)
            covered |= in_box
            new = G[~covered]
            if len(new):
                X = c + new[:, 0:1] * R[:, 0] + new[:, 1:2] * R[:, 1]
                # couleur de la gaussienne assez opaque la plus proche, a l'etat median
                ref = torch.nonzero(alpha > 0.2, as_tuple=True)[0]
                col = torch.full((len(X), 3), 0.5, device=dev)
                if len(ref):
                    d = torch.cdist(X, xyz_m[ref])
                    col = color_m[ref[d.argmin(1)]]
                n = len(X)
                q = torch.tensor(rot_to_quat(w['R']), dtype=torch.float32, device=dev)
                lin = softplus_inv(torch.tensor(max(0.6 * sp - dec.scale_min, 1e-5)))
                lnn = torch.tensor(thickness_raw(w, dec.scale_min))
                t = (0.5 - dec.alpha_min) / (1 - dec.alpha_min)
                add = dict(
                    xyz=X,
                    logs_raw=torch.stack([lin.expand(n), lin.expand(n), lnn.expand(n)], -1).to(dev),
                    quat_raw=q.expand(n, 4).clone(),
                    log_alpha=torch.full((n,), float(np.log(t / (1 - t))), device=dev),
                    color_raw=torch.logit(col.clamp(1e-3, 1 - 1e-3)))
                resize(dec, opt, add=add)
                self.anchor = torch.cat([self.anchor, X])
                tag = torch.full((n,), wid, dtype=torch.long, device=dev)
                self.id = torch.cat([self.id, tag])
                self.born = torch.cat([self.born, tag])
            # decalage normal aleatoire pour toute gaussienne du mur encore exactement dans le
            # plan (nouvelles, nouvellement rattachees, et celles d'avant cette regle)
            c, R, h = frame(w, dev)
            mine = torch.nonzero(self.id == wid, as_tuple=True)[0]
            lz = (self.anchor[mine] - c) @ R[:, 2]
            # 1e-5 et non 1e-7 : a des coordonnees de l'ordre de 3 en float32, l'arrondi seul
            # laisse ~3e-7 hors du plan, et 4 177 gaussiennes echappaient ainsi au decalage
            flat0 = mine[lz.abs() < 1e-5]
            if len(flat0):
                t = JITTER * float(w.get('thickness', THICKNESS))
                off = (torch.rand(len(flat0), device=dev) * 2 - 1) * t
                self.anchor[flat0] += off[:, None] * R[:, 2]
            log('  mur %d : %d gaussiennes rattachees, %d ajoutees (grille de %d points, pas '
                '%.3f, dont %d dans un pave, jamais combles), %d au total sur le mur'
                % (wid, n_near, len(new), len(G), sp, int(in_box.sum()),
                   int((self.id == wid).sum())))
        self.enforce(dec)
