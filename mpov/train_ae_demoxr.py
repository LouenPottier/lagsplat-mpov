"""
AE de la languette : encodeur (image recadree + position du bras) -> q, decodeur (2+5)D GS
parametre par z = (u, q), entraine sur la video ENTIERE.

Trois idees, chacune avec sa loss.

1. RECONSTRUCTION. Le decodeur est le GaussianSplatDecoder2pt de pipeline2 (rendu gsplat),
   avec un latent de dimension 5 : les deux coordonnees MESUREES de l'effecteur (u) et les
   trois coordonnees libres de forme (q). La fenetre etant centree sur le poignet, TOUT dans
   l'image bouge avec u : le fond defile en sens inverse, le bras et la pince sont fixes dans
   ce repere, la languette s'y deforme. D'ou u en entree de tout le monde.

2. q NE SERT QUE DANS LE MASQUE. Hors de l'objet, rien ne doit dependre de q. On penalise
   donc, pour chaque gaussienne hors du masque, sa sensibilite a q :
       - des CENTRES : G_ia = ||d mu_i / d q_a||, forme close (le decodeur est affine en z,
         `latent_jacobian`), sans autograd ;
       - de la PORTE latente w_z : `gate_sensitivity`, second et dernier canal par lequel le
         rendu depend de q. La covariance conditionnelle, elle, ne depend PAS de q
         (propriete du conditionnement gaussien), il n'y a donc rien a y contraindre.
   La ponderation est CONTINUE : chaque gaussienne est pesee par (1 - m), m etant la valeur
   du masque interpolee a sa position courante, et par son opacite x porte (detachees, pour
   que la penalite ne se satisfasse pas en rendant les gaussiennes transparentes).

3. q INDEPENDANT DE u. Sur toute fenetre de --win-sec secondes, la MOYENNE SIMPLE de q doit
   etre nulle. La position du bras varie lentement (echelons et plateaux), donc une composante
   de q qui suivrait u aurait une moyenne locale non nulle ; une oscillation de deformation,
   elle, a une moyenne locale nulle. C'est une moyenne signee, pas une moyenne de valeurs
   absolues : on n'interdit pas a q de bouger, on lui interdit de DERIVER avec u.

Usage : py train_ae_demoxr.py --frames frames_256.npy --masks masks_full.npz
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).parent
PIPE = HERE
sys.path.insert(0, str(PIPE))
# Cadence des MOV mpov : 60000/1001, et non celle d'img5175 (40792/680.376667 = 59.955).
# Elle ne sert ici qu'a convertir des DUREES en nombres de frames (--val-sec, --win-sec),
# jamais a aligner quoi que ce soit, mais autant ne pas trainer la valeur d'un autre cas.
FPS = 60000 / 1001


def load_models():
    import _bootstrap
    cfg = _bootstrap.load_config()            # config de base, on n'en prend que les classes
    cfg.DEC2PT_BACKEND = 'gsplat'
    import models
    import models_2pt
    return cfg, models, models_2pt


class EncoderU(nn.Module):
    """CpAE sur l'image + position du bras concatenee, -> q (d_q)."""

    def __init__(self, models, img_size, d_q, n_feat=32, n_channels=3, whiten=True,
                 n_u=2):
        super().__init__()
        self.cnn = models.CpAEEncoder(img_size, n_feat, n_channels=n_channels,
                                      normalize=False, channels=(32, 64, 128))
        # `n_u` : dimension de l'entree MESUREE. Elle vaut 2 quand on donne la position
        # image de la pince `(u_x, u_y)`, et 1 quand on donne l'abscisse curviligne `s`.
        # `s` suffit et n'est pas redondant : le mouvement du robot est a un parametre, donc
        # `u_y` est une fonction de `u_x` (cf. `curvilinear_s.py`). Donner les deux revient a
        # donner deux fois la meme chose, et laisse au decodeur une direction latente que
        # rien ne contraint.
        self.head = nn.Sequential(nn.Linear(n_feat + n_u, 64), nn.ELU(), nn.Linear(64, d_q))
        # Blanchiment de q (WhiteningLayer du pipeline, cf. ENC_NORMALIZE). Sans lui, les
        # deux penalites (moyenne nulle, purete hors masque) ont un minimum trivial : q -> 0,
        # et le decodeur reconstruit alors tout avec u seul. Le blanchiment impose cov(q) = I,
        # donc q garde une echelle O(1) comme u, et les penalites ne peuvent plus etre
        # satisfaites en rapetissant q.
        # eps petit (1e-8 et non 1e-4) : la ridge PLAFONNE le facteur d'amplification a
        # 1/sqrt(eps), donc avec eps=1e-4 un q d'ecart-type 1e-4 ressortait a 0.01 au lieu
        # de 1, et le blanchiment ne protegeait plus de l'effondrement (mesure au demarrage).
        self.whiten = models.WhiteningLayer(d_q, eps=1e-8) if whiten else None

    def raw(self, img, u):
        """q AVANT blanchiment."""
        return self.head(torch.cat([self.cnn(img), u], 1))

    def forward(self, img, u, extra=None):
        """
        `extra` : q bruts de frames SUPPLEMENTAIRES, encodees mais pas decodees, qui servent
        uniquement a estimer la covariance du blanchiment. Sans elles, cette covariance est
        estimee sur le seul lot rendu (32 frames), donc la transformee change d'un pas a
        l'autre et q(t) bouge sans que l'encodeur ait appris quoi que ce soit. Elles ne
        coutent qu'un passage d'encodeur, pas de rendu.
        """
        q = self.raw(img, u)
        if self.whiten is None:
            return q
        if extra is None or extra.numel() == 0:
            return self.whiten(q)
        n = q.shape[0]
        return self.whiten(torch.cat([q, extra], 0))[:n]


@torch.no_grad()
def init_new_gaussians(dec, K_old, packed, Tm, mshape, ref, S, d_q, dev, seed=0):
    """Initialise les gaussiennes K_old: de `dec`.

    Positions tirees dans l'OCCUPATION du masque (frequence de presence de l'objet, moyennee
    sur les frames disponibles), et non sur la carte de gradient de smart_init, qui repartit
    sur toute la scene et ne met presque rien sur l'objet, petit en pixels.
    """
    n_add = dec.mu_raw.shape[0] - K_old
    if packed is None:
        raise SystemExit('la croissance de gaussiennes tire les positions dans l occupation '
                         'du masque : elle demande --masks, pas --masks none')
    occ = unpack_masks(packed, np.arange(0, Tm, max(1, Tm // 300)), mshape).mean(0)
    pr = (occ / occ.sum()).ravel()
    sel = np.random.default_rng(seed).choice(pr.size, n_add, p=pr)
    yy, xx = np.unravel_index(sel, occ.shape)
    jit = np.random.default_rng(seed + 1).uniform(0, 1, size=(n_add, 2))
    xy = np.stack([(xx + jit[:, 0]) / occ.shape[1], (yy + jit[:, 1]) / occ.shape[0]], 1)
    xyt = torch.tensor(xy, dtype=torch.float32, device=dev).clamp(1e-3, 1 - 1e-3)
    dec.mu_raw[K_old:, :2] = torch.logit((xyt + 0.1) / 1.2)
    ref_hw = ref.permute(1, 2, 0)
    pix = (xyt * torch.tensor([S - 1, S - 1], device=dev)).long()
    col = ref_hw[pix[:, 1], pix[:, 0]].clamp(1e-3, 1 - 1e-3)
    dec.color_raw[K_old:] = torch.logit(col)
    # Centres LATENTS des nouvelles : etales sur la distribution de z (q blanchi, donc
    # N(0,1)). smart_init les mettrait toutes a q = 0, c'est-a-dire toutes centrees sur le
    # meme etat, alors qu'on les ajoute justement pour couvrir les differentes formes.
    dec.mu_raw[K_old:, 2:] = torch.randn(n_add, 2 + d_q, device=dev)
    print('nouvelles gaussiennes : x %.2f..%.2f, y %.2f..%.2f (coord. normalisees)'
          % (xy[:, 0].min(), xy[:, 0].max(), xy[:, 1].min(), xy[:, 1].max()))


def unpack_masks(packed, idx, hw):
    m = np.unpackbits(packed[idx], axis=1)[:, :hw[0] * hw[1]]
    return m.reshape(-1, *hw).astype(np.float32)


@torch.no_grad()
def evaluate(dec, enc, F_mm, U, T_tr, T, a, dev, step, run=HERE, n=16):
    """L1 sur l'entrainement et sur la tranche held-out, + planche image/rendu."""
    import cv2
    out_msg = []
    tiles = []
    for name, lo, hi in (('train', 0, T_tr), ('held-out', T_tr, T)):
        if hi - lo < n:
            continue
        idx = np.linspace(lo, hi - 1, n).astype(int)
        img = torch.tensor(np.asarray(F_mm[idx], np.float32) / 255.0, device=dev)
        q = enc(img, U[idx])
        rend = dec(torch.cat([U[idx], q], 1))
        out_msg.append('%s L1 %.4f' % (name, F.l1_loss(rend, img).item()))
        k = idx[::max(1, n // 4)][:4]
        i2 = torch.tensor(np.asarray(F_mm[k], np.float32) / 255.0, device=dev)
        r2 = dec(torch.cat([U[k], enc(i2, U[k])], 1))
        row = torch.cat([torch.cat([i2[j], r2[j]], dim=1) for j in range(len(k))], dim=2)
        tiles.append((row.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0))
    if tiles:
        h = max(t.shape[0] for t in tiles)
        board = np.concatenate([np.pad(t, ((0, h - t.shape[0]), (0, 0), (0, 0))) for t in tiles], 0)
        cv2.imwrite(str(run / ('recon_%06d.png' % step)),
                    cv2.cvtColor(board, cv2.COLOR_RGB2BGR))

    # q(t) et u(t) sur TOUTE la video, sous-echantillonnes : c'est la que se lit l'effet des
    # deux penalites (q qui suivrait u se verrait tout de suite sous la courbe de u).
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ids = np.arange(0, T, max(1, T // 1500))
    qs = []
    for i in range(0, len(ids), 64):
        k = ids[i:i + 64]
        im = torch.tensor(np.asarray(F_mm[k], np.float32) / 255.0, device=dev)
        qs.append(enc(im, U[k]).cpu().numpy())
    qs = np.concatenate(qs)
    t = ids / FPS
    un = U[ids].cpu().numpy()
    d_q = qs.shape[1]
    # `un` a 2 colonnes quand l'entree mesuree est `(u_x, u_y)` et 1 quand c'est l'abscisse
    # curviligne `s` : une sous-figure par colonne, sans supposer laquelle.
    n_u = un.shape[1]
    lab = ['u_x', 'u_y'] if n_u == 2 else ['s']
    fig, ax = plt.subplots(d_q + n_u, 1, figsize=(15, 1.6 * (d_q + n_u)), sharex=True)
    for k in range(d_q):
        ax[k].plot(t, qs[:, k], lw=.6, color='C%d' % k)
        ax[k].set_ylabel('q%d' % k, fontsize=8)
    for j in range(n_u):
        ax[d_q + j].plot(t, un[:, j], lw=.7, color='k' if j == 0 else '0.4')
        ax[d_q + j].set_ylabel(lab[j], fontsize=8)
    for x in ax:
        x.axvline(T_tr / FPS, color='r', lw=1)
    ax[-1].set_xlabel('t (s)   (trait rouge = debut du held-out)')
    ax[0].set_title('step %d' % step, fontsize=9)
    fig.tight_layout(); fig.savefig(run / ('q_%06d.png' % step), dpi=90); plt.close(fig)
    out_msg.append('q std %s' % np.round(qs.std(0), 2))
    return 'eval step %d : ' % step + ', '.join(out_msg) + ' (haut train, bas held-out)'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', default='frames_256.npy')
    ap.add_argument('--masks', default='masks_full.npz')
    ap.add_argument('--track', default='gripper_corrected.npy',
                    help="suivi de l'effecteur. `gripper_corrected.npy` est le suivi CORRIGE "
                         "(60 recalages manuels, biais vertical du NCC retire) et c'est celui "
                         "avec lequel les crops ont ete regeneres. Tous les scripts du cas "
                         "doivent utiliser le MEME : entrainer avec un suivi et evaluer avec "
                         "l'autre donne un rendu flou et dedouble, sans qu'aucune erreur ne "
                         "soit levee (mesure : L1 0.019 contre 0.047 sur les memes poids).")
    ap.add_argument('--d-q', type=int, default=3)
    ap.add_argument('--gaussians', type=int, default=20000)
    ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--n-win', type=int, default=4, help='fenetres par lot')
    ap.add_argument('--micro-win', type=int, default=0,
                    help='fenetres par SOUS-LOT (0 = pas de decoupage). Le lot est traite en '
                         'plusieurs passes avec accumulation de gradient, ce qui permet un '
                         'grand lot sans saturer la memoire. Le decoupage se fait par '
                         'FENETRES ENTIERES : la loss de moyenne nulle porte sur des fenetres '
                         'completes, donc elle garde exactement le meme sens.')
    ap.add_argument('--n-frame', type=int, default=8, help='frames tirees par fenetre')
    ap.add_argument('--win-sec', type=float, default=2.0)
    ap.add_argument('--lr-dec', type=float, default=2e-3)
    ap.add_argument('--lr-enc', type=float, default=1e-4)
    ap.add_argument('--w-zero', type=float, default=1.0, help='poids de la moyenne nulle de q')
    ap.add_argument('--w-outside', type=float, default=1.0, help='poids de la purete hors masque')
    ap.add_argument('--w-gate', type=float, default=0.1, help='part de la porte dans (2)')
    ap.add_argument('--w-kl', type=float, default=0.005,
                    help='poids de la KL agregee vers N(0,I) (terme lam_beta de Krauss et al. '
                         '2026, dont la valeur officielle est 0.005 ; leur KL est celle d un '
                         'VAE par echantillon, la notre celle de la distribution agregee du '
                         'lot, donc les echelles ne sont pas strictement comparables)')
    ap.add_argument('--val-sec', type=float, default=60.0, help='derniere tranche en held-out')
    ap.add_argument('--no-whiten', action='store_true', help='desactive le blanchiment de q')
    ap.add_argument('--w-scale', type=float, default=0.0,
                    help="ancrage d'echelle (std(q) -> 1), par composante. Necessaire des que "
                         "le blanchiment est retire : la moyenne nulle et la purete hors "
                         "masque ont toutes deux un minimum trivial en q -> 0, et la "
                         "reconstruction seule ne le retient pas (le decodeur compense en "
                         "amplifiant d(mu)/dq). Mesure : sans lui, q std tombe de 1.0 a 0.09 "
                         "en 500 pas. Contrairement au blanchiment, il ne force AUCUNE "
                         "decorrelation, donc il ne replie pas la variete.")
    ap.add_argument('--whiten-extra', type=int, default=0,
                    help="frames encodees EN PLUS du lot pour estimer la covariance du "
                         "blanchiment (elles ne sont pas rendues). DEFAUT 0, et il faut de "
                         "tres bonnes raisons d'y toucher : ces frames sont encodees sous "
                         "no_grad, donc la transformee de blanchiment cesse de dependre du "
                         "lot AU SENS DU GRADIENT. Or c'est par cette dependance que le "
                         "blanchiment s'oppose a l'effondrement du latent : ecraser une "
                         "direction se paie tout de suite quand la transformee suit le lot. "
                         "Mesure a 96 : les deux composantes de q deviennent colineaires a "
                         "0.99999 (rapport de valeurs propres 3.7e6) contre 0.48 a 0.")
    ap.add_argument('--freeze-enc', action='store_true',
                    help='encodeur GELE : q(t) fige, seul le decodeur apprend')
    ap.add_argument('--init', default=None, help='checkpoint de depart (reprise)')
    ap.add_argument('--shift-xy', type=float, nargs=2, default=[0.0, 0.0],
                    help='translation (dx, dy) en px de la fenetre 640 a appliquer aux '
                         'gaussiennes du checkpoint, quand le recadrage a change')
    ap.add_argument('--add-gauss', type=int, default=0,
                    help='gaussiennes AJOUTEES au checkpoint, tirees dans la zone du masque')
    ap.add_argument('--grow-every', type=int, default=0,
                    help='ajoute --grow-n gaussiennes tous les N pas (0 = jamais), dans le '
                         'masque comme --add-gauss. Les moments d Adam des anciennes sont '
                         'conserves, les nouvelles partent a zero.')
    ap.add_argument('--grow-n', type=int, default=10000)
    ap.add_argument('--grow-until', type=int, default=0,
                    help='dernier pas ou l on ajoute encore (0 = pas de limite)')
    ap.add_argument('--save-at', type=int, nargs='*', default=[],
                    help='pas ou l on garde une COPIE du checkpoint (ae_<pas>.pt)')
    ap.add_argument('--u-sigma', type=float, default=4.0,
                    help="lissage temporel gaussien de u, en frames. 0 pour un suivi deja "
                         "fonction de l image (gripper_curve.npy) : lisser decale le bras "
                         "jusqu'a 12 px proxy autour des echelons (u_manifold/test_same_u.py).")
    ap.add_argument('--mask-offset', type=int, default=0,
                    help='frame correspondant au masque 0 (test sur un extrait)')
    ap.add_argument('--out', default='ae')
    ap.add_argument('--plot-every', type=int, default=500)
    a = ap.parse_args()

    cfg, models, models_2pt = load_models()
    dev = 'cuda'
    run = HERE / 'runs' / a.out          # tout le run (poids, planches, log) dans son dossier
    run.mkdir(parents=True, exist_ok=True)
    F_mm = np.load(HERE / a.frames, mmap_mode='r')
    T, C, S, _ = F_mm.shape
    # Masques facultatifs. `--masks none` entraine SANS la penalite de purete hors masque,
    # ce qui sert a demarrer un cas dont les masques SAM 2 ne sont pas encore faits. La
    # reconstruction et la moyenne nulle de q, elles, n'en ont pas besoin. A ne pas garder :
    # sans masque, rien n'empeche q de servir a affiner le rendu du bras au lieu de decrire
    # la deformation.
    if str(a.masks).lower() in ('none', ''):
        packed, mshape, Tm = None, (1, 1), 0
        print('AUCUN MASQUE : la penalite de purete hors masque est desactivee.')
    else:
        z = np.load(HERE / a.masks)
        packed, mshape = z['packed'], tuple(z['shape'][1:])
        Tm = int(z['shape'][0])
    off = a.mask_offset
    # Masques PARTIELS admis (SAM 2 tourne encore) : les frames non couvertes servent quand
    # meme a la reconstruction et a la moyenne nulle de q, seule la penalite de dependance a
    # q est omise pour elles, faute de savoir ou est l'objet.
    u_raw = np.load(HERE / a.track)[:T].astype(np.float32)
    if u_raw.ndim == 1:
        u_raw = u_raw[:, None]
    u_raw = u_raw[:, :2]
    n_u = u_raw.shape[1]
    from scipy.ndimage import gaussian_filter1d
    if a.u_sigma > 0:
        u_raw = gaussian_filter1d(u_raw, a.u_sigma, axis=0, mode='nearest')
    u_mean, u_std = u_raw.mean(0), u_raw.std(0) + 1e-6
    U = torch.tensor((u_raw - u_mean) / u_std, device=dev)
    print('T=%d frames, image %d px, masques %s, u normalise (std %s)'
          % (T, S, mshape, np.round(u_std, 1)))

    f_lo, f_hi = off, off + Tm              # plage couverte par les masques
    print('masques disponibles pour les frames %d..%d (%.1f %% de la video)'
          % (f_lo, f_hi - 1, 100 * Tm / T))
    n_val = int(a.val_sec * FPS)
    T_tr = T - n_val
    W = int(a.win_sec * FPS)

    ck = None
    K_old = 0
    n_gauss = a.gaussians
    if a.init:
        ck = torch.load(HERE / a.init, map_location=dev, weights_only=False)
        K_old = ck['dec']['mu_raw'].shape[0]
        n_gauss = K_old + a.add_gauss
        print('reprise : %d gaussiennes du checkpoint + %d nouvelles dans le masque'
              % (K_old, a.add_gauss))
    dec = models_2pt.build_decoder2pt(n_u + a.d_q, n_gauss, (S, S), n_channels=C,
                                      backend='gsplat').to(dev)
    enc = EncoderU(models, (S, S), a.d_q, n_channels=C, whiten=not a.no_whiten,
                   n_u=n_u).to(dev)
    ref = torch.tensor(F_mm[0].astype(np.float32) / 255.0, device=dev)
    sub = np.random.default_rng(0).choice(np.arange(f_lo, T_tr), min(2000, T_tr - f_lo),
                                          replace=False)
    z_all = torch.cat([U[sub], torch.zeros(len(sub), a.d_q, device=dev)], 1)
    dec.smart_init(ref, z_all, sigma_xy_scale=6.0 / S, sigma_z_scale=1.0, sigma_z_floor=0.5)
    if ck is not None:
        enc_sd = ck['enc']
        if a.no_whiten and any(k.startswith('whiten.') for k in enc_sd):
            # Le blanchiment est une transformee AFFINE q_w = (q - m) W. On l'absorbe dans la
            # derniere couche lineaire de la tete au lieu de le supprimer : sans cela, q
            # changerait brutalement d'echelle et d'orientation au moment du basculement, et
            # le decodeur, entraine sur l'ancien q, rendrait n'importe quoi. Ainsi q(t) est
            # INCHANGE a l'instant du basculement, et libre ensuite.
            m = enc_sd['whiten.running_mean']
            # NB : pas `W`, qui est deja la longueur de la fenetre de la loss de moyenne nulle
            Wh = enc_sd['whiten.running_W']
            kw = [k for k in enc_sd if k.startswith('head.')]
            last_w = max(k for k in kw if k.endswith('.weight'))
            last_b = last_w[:-6] + 'bias'
            Wt = Wh.t().to(enc_sd[last_w].dtype)
            enc_sd = {k: v for k, v in enc_sd.items() if not k.startswith('whiten.')}
            enc_sd[last_w] = Wt @ enc_sd[last_w]
            enc_sd[last_b] = Wt @ (enc_sd[last_b] - m.to(enc_sd[last_b].dtype))
            print('blanchiment absorbe dans %s : q(t) inchange au basculement' % last_w)
        enc.load_state_dict(enc_sd)
        with torch.no_grad():
            for name in ('mu_raw', 'L_raw', 'log_alpha', 'color_raw'):
                getattr(dec, name)[:K_old] = ck['dec'][name]
            if any(a.shift_xy):
                # Le recadrage a bouge : on translate les centres pour repartir de la bonne
                # place. mu_xy = sigmoid(raw)*1.2 - 0.1, donc on translate dans l'espace
                # des coordonnees, pas dans celui des logits.
                sx = torch.tensor(a.shift_xy, dtype=torch.float32, device=dev) / 640.0
                xy = (torch.sigmoid(dec.mu_raw[:K_old, :2]) * 1.2 - 0.1) + sx
                dec.mu_raw[:K_old, :2] = torch.logit(
                    ((xy + 0.1) / 1.2).clamp(1e-4, 1 - 1e-4))
                print('gaussiennes translatees de (%.2f, %.2f) px' % tuple(a.shift_xy))
            if a.add_gauss > 0:
                init_new_gaussians(dec, K_old, packed, Tm, mshape, ref, S, a.d_q, dev, seed=0)
        print('reprise depuis', a.init)
    if a.freeze_enc:
        # eval() ET requires_grad=False : le mode eval fait passer la WhiteningLayer sur ses
        # running stats, donc q(t) devient une fonction FIXE des images, identique d'un pas a
        # l'autre. Sans lui, le blanchiment se recalculerait sur chaque batch et q bougerait
        # encore, alors qu'on veut justement le figer.
        enc.eval()
        for p_ in enc.parameters():
            p_.requires_grad_(False)
        groups = [{'params': dec.parameters(), 'lr': a.lr_dec}]
        print('encodeur GELE (eval + requires_grad=False)')
    else:
        groups = [{'params': dec.parameters(), 'lr': a.lr_dec},
                  {'params': enc.parameters(), 'lr': a.lr_enc}]
    opt = torch.optim.Adam(groups)
    rng = np.random.default_rng(0)
    log = open(run / 'log.txt', 'a')
    t0 = time.time()
    hist = []
    mw = a.micro_win if a.micro_win > 0 else a.n_win
    n_chunk = int(np.ceil(a.n_win / mw))
    if n_chunk > 1:
        print('lot de %d fenetres traite en %d sous-lots de %d (accumulation de gradient)'
              % (a.n_win, n_chunk, mw))
    for step in range(1, a.steps + 1):
        if (a.grow_every > 0 and step > 1 and (step - 1) % a.grow_every == 0
                and (a.grow_until <= 0 or step - 1 <= a.grow_until)):
            # Croissance : nouveau decodeur plus grand, anciennes gaussiennes recopiees, et
            # moments d'Adam des anciennes conserves (les nouvelles partent de zero).
            K_cur = dec.mu_raw.shape[0]
            names = ('mu_raw', 'L_raw', 'log_alpha', 'color_raw')
            new = models_2pt.build_decoder2pt(n_u + a.d_q, K_cur + a.grow_n, (S, S),
                                              n_channels=C, backend='gsplat').to(dev)
            with torch.no_grad():
                new.smart_init(ref, z_all, sigma_xy_scale=6.0 / S, sigma_z_scale=1.0,
                               sigma_z_floor=0.5)
                for name in names:
                    getattr(new, name)[:K_cur] = getattr(dec, name)
            init_new_gaussians(new, K_cur, packed, Tm, mshape, ref, S, a.d_q, dev, seed=step)
            old_state = {n_: opt.state.get(getattr(dec, n_), {}) for n_ in names}
            enc_state = {p_: opt.state[p_] for g in opt.param_groups[1:] for p_ in g['params']
                         if p_ in opt.state}
            dec = new
            groups[0]['params'] = dec.parameters()
            opt = torch.optim.Adam([dict(g, params=list(g['params'])) for g in groups])
            for p_, st in enc_state.items():
                opt.state[p_] = st
            for n_ in names:
                p_ = getattr(dec, n_)
                ns = {}
                for k, v in old_state[n_].items():
                    if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == K_cur:
                        z_ = torch.zeros_like(p_)
                        z_[:K_cur] = v
                        ns[k] = z_
                    else:
                        ns[k] = v
                if ns:
                    opt.state[p_] = ns
            msg = 'step %d : +%d gaussiennes -> %d' % (step - 1, a.grow_n, K_cur + a.grow_n)
            print(msg, flush=True); log.write(msg + '\n'); log.flush()
        starts_all = rng.integers(0, max(1, T_tr - W), size=a.n_win)
        opt.zero_grad(set_to_none=True)
        acc = np.zeros(5)
        for ci in range(n_chunk):
            starts = starts_all[ci * mw:(ci + 1) * mw]
            n_w = len(starts)
            if n_w == 0:
                continue
        # tri PAR FENETRE (pas global) : le lot reste groupe par fenetre, sinon la moyenne
        # de q ci-dessous melangerait des frames de fenetres differentes
            idx = np.concatenate([np.sort(s + rng.choice(W, a.n_frame, replace=False))
                                  for s in starts])
            idx = np.clip(idx, 0, T_tr - 1)
            img = torch.tensor(np.asarray(F_mm[idx], np.float32) / 255.0, device=dev)
            cov = (idx >= f_lo) & (idx < f_hi)          # frames dont on a le masque
            if cov.any():
                mc = torch.tensor(unpack_masks(packed, idx[cov] - off, mshape), device=dev)
                mc = F.interpolate(mc.unsqueeze(1), size=(S, S), mode='bilinear',
                                   align_corners=False)
            cov_t = torch.tensor(cov, device=dev)
            u = U[idx]
            if a.whiten_extra > 0 and not a.freeze_enc and not a.no_whiten:
                xi = rng.integers(0, T_tr, size=a.whiten_extra)
                ximg = torch.tensor(np.asarray(F_mm[np.sort(xi)], np.float32) / 255.0,
                                    device=dev)
                with torch.no_grad():
                    extra = enc.raw(ximg, U[np.sort(xi)])
            else:
                extra = None
            q = enc(img, u, extra)
            zq = torch.cat([u, q], 1)
            out = dec(zq)
            rec = F.l1_loss(out, img)

            # (3) moyenne nulle de q par fenetre
            qw = q.view(n_w, a.n_frame, -1).mean(1)
            zero = qw.pow(2).mean()

            # (2) purete hors masque : sensibilite a q ponderee par (1 - masque)
            A = dec.latent_jacobian()                   # (K, 2, n_u+d_q)
            G = A[:, :, n_u:].norm(dim=1).sum(-1)       # (K,) sensibilite des centres a q
            gs = dec.gate_sensitivity(zq)[:, n_u:].sum(-1)  # (K,) sensibilite de la porte a q
            if cov.any():
                with torch.no_grad():
                    zc = zq[cov_t]
                    mu_c, _, _, opac, _ = dec.render_params(zc)      # (b,K,2), (b,K)
                    p = (mu_c * 2 - 1).clamp(-1.5, 1.5).unsqueeze(2)  # (b,K,1,2) grille [-1,1]
                    mval = F.grid_sample(mc, p, align_corners=False).squeeze(1).squeeze(-1)
                    w = opac * (1.0 - mval)
                    w = w / w.sum().clamp(min=1e-9)
                outside = (w.sum(0) * (G + a.w_gate * gs)).sum()
            else:
                outside = G.sum() * 0.0

            scale = ((q.std(0) - 1.0) ** 2).mean() if (a.no_whiten and a.w_scale > 0) \
                else torch.zeros((), device=dev)

        # KL de la distribution latente AGREGEE du lot vers N(0, I), terme anti-collapse de
        # Krauss et al. 2026 (lam_beta de leur Eq. 5), dans la variante a encodeur
        # DETERMINISTE utilisee par train_all du pipeline :
        #     KL = 1/2 [ tr(Sigma) + mu' mu - d - log det Sigma ]
        # Le log det diverge quand une direction s'ecrase, donc ce terme s'oppose a
        # l'effondrement de q sans imposer, comme le blanchiment, que cov(q) soit exactement I.
            if a.w_kl > 0:
                mq = q.mean(0)
                qc = q - mq
                Sg = qc.T @ qc / max(q.shape[0] - 1, 1) + 1e-6 * torch.eye(a.d_q, device=dev)
                kl = 0.5 * (torch.diagonal(Sg).sum() + mq.pow(2).sum() - a.d_q
                            - torch.logdet(Sg))
            else:
                kl = torch.zeros((), device=dev)

            loss = (rec + a.w_zero * zero + a.w_outside * outside + a.w_scale * scale
                    + a.w_kl * kl)
            (loss * (n_w / a.n_win)).backward()     # moyenne sur le lot complet
            acc += (n_w / a.n_win) * np.array([loss.item(), rec.item(), zero.item(),
                                               outside.item(), kl.item()])
        torch.nn.utils.clip_grad_norm_(list(dec.parameters()) + list(enc.parameters()), 5.0)
        opt.step()
        hist.append([step, acc[1], acc[2], acc[3], acc[4]])
        if step % 50 == 0:
            with torch.no_grad():
                inside = (G * torch.sigmoid(dec.log_alpha)).sum() / torch.sigmoid(dec.log_alpha).sum()
            # `acc` est la moyenne sur le LOT COMPLET (tous les sous-lots), pas sur le
            # dernier ; `q std` reste celui du dernier sous-lot, faute de garder les q.
            msg = ('step %6d  L1 %.4f  zero %.4f  hors-masque %.4f  KL %.3f  |G| moyen %.4f  '
                   'q std %s  %.0f s' % (step, acc[1], acc[2], acc[3], acc[4],
                                         inside.item(),
                                         np.round(q.std(0).detach().cpu().numpy(), 2),
                                         time.time() - t0))
            print(msg, flush=True); log.write(msg + '\n'); log.flush()
        if step % a.plot_every == 0 or step == a.steps:
            torch.save({'dec': dec.state_dict(), 'enc': enc.state_dict(),
                        'u_mean': u_mean, 'u_std': u_std, 'args': vars(a)},
                       run / 'ae.pt')
            np.save(run / 'hist.npy', np.array(hist))
            if step in a.save_at:
                import shutil
                shutil.copy(run / 'ae.pt', run / ('ae_%d.pt' % step))
            vmsg = evaluate(dec, enc, F_mm, U, T_tr, T, a, dev, step, run=run)
            print(vmsg, flush=True); log.write(vmsg + '\n'); log.flush()
    log.close()
    print('fini en %.0f s' % (time.time() - t0))


if __name__ == '__main__':
    main()
