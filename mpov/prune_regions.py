"""
Zones d'elagage dessinees a la main : un lasso (courbe fermee) trace sur l'ecran depuis un
point de vue definit un volume, la partie du frustum de cette camera qui se projette dans le
lasso, sur TOUTE la profondeur. Toute gaussienne dont le centre tombe dans l'un de ces volumes
est a supprimer.

Module partage par l'outil de dessin (prune_tool.py) et par l'entrainement
(train_dec3d_shared.py --prune-regions), pour que les deux appliquent EXACTEMENT le meme test.

Format du fichier (JSON) : {"regions": [{"E": 4x4 monde -> camera (convention OpenCV,
x a droite, y vers le bas, z devant), "K": 3x3, "poly": [[u, v], ...] en pixels de cette
camera, "near": 0.01, "note": ...}, ...]}
"""
import json
from pathlib import Path

import torch


def load_regions(path):
    p = Path(path)
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding='utf-8')).get('regions', [])


def save_regions(path, regions):
    Path(path).write_text(json.dumps({'regions': regions}, indent=1), encoding='utf-8')


def point_in_poly(u, v, poly):
    """Test de parite (lancer de rayon), vectorise. u, v (N,), poly (M, 2) -> (N,) bool."""
    x1, y1 = poly[:, 0], poly[:, 1]
    x2, y2 = poly.roll(-1, 0)[:, 0], poly.roll(-1, 0)[:, 1]
    u, v = u[:, None], v[:, None]
    cross = ((y1 > v) != (y2 > v)) & (u < (x2 - x1) * (v - y1) / (y2 - y1 + 1e-12) + x1)
    return (cross.sum(1) % 2) == 1


def inside_region(xyz, reg):
    """xyz (K, 3) torch -> (K,) bool : centre dans le volume de la zone `reg`."""
    dev, dt = xyz.device, xyz.dtype
    E = torch.tensor(reg['E'], device=dev, dtype=dt)
    K = torch.tensor(reg['K'], device=dev, dtype=dt)
    poly = torch.tensor(reg['poly'], device=dev, dtype=dt)
    Xc = xyz @ E[:3, :3].T + E[:3, 3]
    z = Xc[:, 2]
    ok = z > float(reg.get('near', 0.01))
    zs = z.clamp(min=1e-6)
    u = K[0, 0] * Xc[:, 0] / zs + K[0, 2]
    v = K[1, 1] * Xc[:, 1] / zs + K[1, 2]
    out = torch.zeros_like(ok)
    if ok.any():
        idx = torch.nonzero(ok, as_tuple=True)[0]
        for i in range(0, len(idx), 65536):          # borne la memoire du test (N x M)
            j = idx[i:i + 65536]
            out[j] = point_in_poly(u[j], v[j], poly)
    return out


def inside_regions(xyz, regions):
    """(K, 3) -> (K,) bool : dans AU MOINS une zone."""
    m = torch.zeros(xyz.shape[0], dtype=torch.bool, device=xyz.device)
    for r in regions:
        m |= inside_region(xyz, r)
    return m
