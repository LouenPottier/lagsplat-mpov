"""
Fige l'encodeur et remplace le blanchiment par lot par une normalisation DEFINITIVE,
calculee une fois sur toute la base.

Pourquoi. Le blanchiment par lot est invariant d'echelle, donc la sortie brute de l'encodeur
peut deriver librement : mesure sur le checkpoint d = 2, elle etait tombee a un ecart-type de
0.0086 alors que le blanchiment par lot la renormalisait a 1. Les statistiques accumulees
(moyenne mobile) retardaient sur cette derive, si bien qu'en mode evaluation q sortait 35 fois
trop petit. Tant que l'encodeur bouge, ce n'est qu'un desagrement ; pour figer q(t) une fois
pour toutes, il faut une transformee FIXE et coherente.

Ce que fait ce script :
  1. encode la sortie BRUTE de l'encodeur sur tout le jeu (sous-echantillonne a --stride) ;
  2. calcule mu et W = V diag(lambda^-1/2) V^T sur ces statistiques globales (meme formule
     que `models.WhiteningLayer._global_W`, donc meme convention que ce que le decodeur a vu
     a l'entrainement) ;
  3. ABSORBE (mu, W) dans la derniere couche lineaire de la tete, et supprime la couche de
     blanchiment. L'encodeur devient une fonction fixe, sans statistique mobile, et q(t) ne
     depend plus du lot ni du mode entrainement/evaluation ;
  4. sauve un checkpoint utilisable directement avec --no-whiten --freeze-enc.

Usage : py freeze_encoder.py --ckpt runs/ae_d2/ae.pt --out runs/ae_d2_frozen/ae.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='runs/ae_d2/ae.pt')
    ap.add_argument('--frames', default='frames_256.npy')
    ap.add_argument('--track', default='gripper_corrected.npy')
    ap.add_argument('--stride', type=int, default=5)
    ap.add_argument('--u-sigma', type=float, default=4.0, help='comme train_ae_demoxr.py')
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--eps', type=float, default=1e-8)
    ap.add_argument('--out', default='runs/ae_d2_frozen/ae.pt')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    import _bootstrap
    _bootstrap.load_config()
    import models
    from train_ae_demoxr import EncoderU

    ck = torch.load(HERE / a.ckpt, map_location=dev, weights_only=False)
    args = ck['args']
    d_q = args['d_q']
    F = np.load(HERE / a.frames, mmap_mode='r')
    T, C, S, _ = F.shape
    # Le suivi peut etre a 1 colonne (abscisse curviligne `s`) ou a 2 (`u_x, u_y`). La
    # dimension doit etre celle de l'entrainement, sans quoi la tete de l'encodeur ne se
    # recharge pas ; on la relit donc du checkpoint plutot que de la supposer.
    u = np.load(HERE / a.track)[:T].astype(np.float32)
    if u.ndim == 1:
        u = u[:, None]
    n_u = int(np.atleast_1d(ck['u_mean']).size)
    u = u[:, :n_u]
    if a.u_sigma > 0:
        u = gaussian_filter1d(u, a.u_sigma, axis=0, mode='nearest')
    U = torch.tensor((u - ck['u_mean']) / ck['u_std'], device=dev)
    enc = EncoderU(models, (S, S), d_q, n_channels=C, whiten=True, n_u=n_u).to(dev)
    enc.load_state_dict(ck['enc'])
    enc.eval()

    idx = np.arange(0, T, a.stride)
    raw = np.empty((len(idx), d_q), np.float64)
    with torch.no_grad():
        for i in range(0, len(idx), a.batch):
            k = idx[i:i + a.batch]
            img = torch.tensor(np.asarray(F[k], np.float32) / 255.0, device=dev)
            raw[i:i + len(k)] = enc.raw(img, U[k]).double().cpu().numpy()
            if (i // a.batch) % 40 == 0:
                print('  %d / %d' % (i, len(idx)), flush=True)
    mu = raw.mean(0)
    cov = np.cov((raw - mu).T).reshape(d_q, d_q) + a.eps * np.eye(d_q)
    lam, V = np.linalg.eigh(cov)
    W = V * (1.0 / np.sqrt(np.maximum(lam, a.eps)))[None, :]     # comme _global_W
    print('sur %d frames : q brut std %s' % (len(idx), np.round(raw.std(0), 5)))
    print('valeurs propres de la covariance : %s' % np.round(lam, 8))
    qn = (raw - mu) @ W
    print('apres normalisation definitive : std %s, correlation %.4f'
          % (np.round(qn.std(0), 4), np.corrcoef(qn.T)[0, 1]))

    # Absorption dans la derniere couche : q_norm = (A h + b - mu) W  =>  A' = W^T A,
    # b' = W^T (b - mu). L'encodeur sort alors q_norm directement.
    sd = {k: v for k, v in ck['enc'].items() if not k.startswith('whiten.')}
    last_w = max(k for k in sd if k.startswith('head.') and k.endswith('.weight'))
    last_b = last_w[:-6] + 'bias'
    Wt = torch.tensor(W.T, dtype=sd[last_w].dtype, device=sd[last_w].device)
    mt = torch.tensor(mu, dtype=sd[last_b].dtype, device=sd[last_b].device)
    sd[last_w] = Wt @ sd[last_w]
    sd[last_b] = Wt @ (sd[last_b] - mt)
    args = dict(args); args['no_whiten'] = True; args['freeze_enc'] = True
    out = HERE / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'dec': ck['dec'], 'enc': sd, 'u_mean': ck['u_mean'], 'u_std': ck['u_std'],
                'args': args, 'whiten_mu': mu, 'whiten_W': W}, out)

    # Controle : le nouvel encodeur, SANS couche de blanchiment, doit reproduire qn.
    enc2 = EncoderU(models, (S, S), d_q, n_channels=C, whiten=False, n_u=n_u).to(dev)
    enc2.load_state_dict(sd)
    enc2.eval()
    with torch.no_grad():
        k = idx[:a.batch]
        img = torch.tensor(np.asarray(F[k], np.float32) / 255.0, device=dev)
        chk = enc2(img, U[k]).double().cpu().numpy()
    err = np.abs(chk - qn[:len(k)]).max()
    print('controle : ecart max au calcul direct = %.2e' % err)
    print('ecrit', out)


main()
