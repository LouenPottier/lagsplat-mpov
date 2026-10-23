"""
Applique a un checkpoint ZonedSplat (train_dec3d_zoned.py) les zones d'elagage dessinees avec
`prune_tool.py --ckpt` : les gaussiennes designees sont SUPPRIMEES du checkpoint.

Test d'une zone. Chaque zone enregistre l'etat (s, q) affiche quand elle a ete tracee. Une
gaussienne est supprimee si son centre, a CET etat, est dans le volume de la zone (frustum du
point de vue qui se projette dans le lasso, cf. prune_regions.py). Pour les gaussiennes fixes,
l'etat ne change rien ; pour celles des zones s et q, c'est exactement ce que l'outil montrait
en vert au moment de valider. L'outil et ce script appellent la meme fonction (`zone_mask`).

Gaussiennes designees UNE PAR UNE (clic dans l'outil) : indices dans
<dossier>/<stem>_prune_ids.json, supprimees en plus des zones. Ces indices sont ceux du checkpoint
d'origine : ils ne valent que pour lui.

Suppression. Champs par gaussienne (xyz, quat, log_scale, logit_alpha, sh0, shN, carrier_w,
cov_floor) filtres ; idx_s / idx_q renumerotes en gardant leur ordre ; lignes de tete des champs
(12 par gaussienne, dans l'ordre de idx_s / idx_q) et q_gate filtrees de meme.

Usage :
    py apply_prune_zoned.py --ckpt runs/dec3d_zoned_low20k_c05/dec.pt
        (zones : <dossier>/<stem>_prune_regions.json ; sortie : <dossier>/<stem>_pruned.pt)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from prune_regions import load_regions, inside_region

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))
N_OUT = 12                                   # sorties de tete par gaussienne (field_s, field_q)


def regions_path_of(ckpt):
    ckpt = Path(ckpt)
    return ckpt.with_name(ckpt.stem + '_prune_regions.json')


def ids_path_of(ckpt):
    ckpt = Path(ckpt)
    return ckpt.with_name(ckpt.stem + '_prune_ids.json')


def load_ids(path):
    p = Path(path)
    return json.loads(p.read_text(encoding='utf-8')).get('ids', []) if p.exists() else []


def save_ids(path, ids):
    Path(path).write_text(json.dumps({'ids': [int(i) for i in ids]}), encoding='utf-8')


def z_median(Zall):
    return np.median(Zall, 0).astype(np.float64)


@torch.no_grad()
def zone_mask(dec, regions, z_of):
    """(K,) bool sur le device du decodeur : centre dans au moins une zone, chaque zone testee
    a l'etat ou elle a ete tracee. `dec` = ZonedAdapter (demo3d.py), z_of(s_raw, q) -> z (3,)."""
    m = None
    for r in regions:
        st = r.get('state') or {}
        z = z_of(st['s'], st.get('q')) if 's' in st else z_of(None, None)
        zt = torch.tensor(z[None], dtype=torch.float32, device='cuda')
        xyz = dec.state(zt)[0][0]
        mi = inside_region(xyz, r)
        m = mi if m is None else (m | mi)
    if m is None:
        m = torch.zeros(dec.n_gaussians, dtype=torch.bool, device='cuda')
    return m


def drop(d, remove):
    """Etat du decodeur `d` (dict) sans les gaussiennes `remove` (K,) bool, cpu."""
    K = d['xyz'].shape[0]
    keep = ~remove
    old2new = torch.full((K,), -1, dtype=torch.long)
    old2new[keep] = torch.arange(int(keep.sum()))
    out = dict(d)
    for k, v in d.items():
        if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == K and k not in ('idx_s', 'idx_q'):
            out[k] = v[keep].clone()
    for zone, field, extra in (('idx_s', 'field_s', ()), ('idx_q', 'field_q', ('q_gate',))):
        idx = d[zone].long()
        kp = keep[idx]                                       # positions gardees dans la zone
        out[zone] = old2new[idx[kp]].to(d[zone].dtype)
        rows = (torch.nonzero(kp, as_tuple=True)[0][:, None] * N_OUT
                + torch.arange(N_OUT)[None]).reshape(-1)
        for p in ('weight', 'bias'):
            key = '%s.head.%s' % (field, p)
            assert d[key].shape[0] == len(idx) * N_OUT, key
            out[key] = d[key][rows].clone()
        for e in extra:
            out[e] = d[e][kp].clone()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--regions', default=None, help='defaut : <dossier>/<stem>_prune_regions.json')
    p.add_argument('--out', default=None, help='defaut : <dossier>/<stem>_pruned.pt')
    a = p.parse_args()
    from demo3d import load_decoder
    src = HERE / a.ckpt
    ck = torch.load(src, map_location='cuda', weights_only=False)
    ar = ck['args']
    regions = load_regions(Path(a.regions) if a.regions else regions_path_of(src))
    ids = load_ids(ids_path_of(src))
    if not regions and not ids:
        raise SystemExit('aucune zone ni gaussienne designee')
    dec, _ = load_decoder(ck)
    Zall = np.load(HERE / ar['z_all'])
    ae = torch.load(HERE / (ar.get('init_ae') or 'runs/i1_d2_frozen/ae.pt'), map_location='cpu',
                    weights_only=False)
    um, us = float(np.atleast_1d(ae['u_mean'])[0]), float(np.atleast_1d(ae['u_std'])[0])

    def z_of(s_raw, q):
        z = z_median(Zall)
        if s_raw is not None:
            z[0] = (s_raw - um) / us
        if q is not None:
            z[1:1 + len(q)] = q
        return z

    rm = zone_mask(dec, regions, z_of).cpu()
    n_zone = int(rm.sum())
    if ids:
        rm[torch.tensor(ids, dtype=torch.long)] = True
    d = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in ck['dec'].items()}
    K = d['xyz'].shape[0]
    ins = torch.zeros(K, dtype=torch.bool)
    ins[d['idx_s'].long()] = True
    inq = torch.zeros(K, dtype=torch.bool)
    inq[d['idx_q'].long()] = True
    out = dict(ck)
    out['dec'] = drop(d, rm)
    n = out['dec']
    out['n'] = dict(q=len(n['idx_q']), s=len(n['idx_s']),
                    fixed=n['xyz'].shape[0] - len(n['idx_q']) - len(n['idx_s']))
    out['note_prune'] = ('%d zones de %s (%d gaussiennes) et %d gaussiennes designees : %d '
                         'supprimees (fixes %d, zone s %d, zone q %d), %d -> %d'
                         % (len(regions), a.regions or regions_path_of(src).name, n_zone,
                            len(ids), int(rm.sum()),
                            int((rm & ~ins & ~inq).sum()), int((rm & ins).sum()),
                            int((rm & inq).sum()), K, n['xyz'].shape[0]))
    dst = HERE / a.out if a.out else src.with_name(src.stem + '_pruned.pt')
    torch.save(out, dst)
    print(out['note_prune'])
    print('ecrit', dst)


if __name__ == '__main__':
    main()
