"""
Decodeur 3D a deformation PAR ZONES, initialise sur la 3DGS statique de la scene fixe.

Point de depart : `runs/static3dgs_v3/scene.pt` (3DGS statique de la video de scene fixe de
l'iphone3, harmoniques spheriques de degre 3, repere de la piece). Ses gaussiennes sont
reparties en trois groupes par les paves de `runs/static_i3/zones.json` (`zones_static.py`),
une fois pour toutes, a leur position actuelle :

  - pave q (sous l'axe M12M10) : dependent de s ET de q ;
  - pave s prive du pave q : dependent de s SEULEMENT ;
  - ailleurs : FIXES, independantes de l'etat, mais optimisees (on affine la 3D).

La dependance est STRUCTURELLE : deux champs `SharedDeformField` distincts, l'un ne recoit
que s, l'autre (s, q). Pas de penalite de sensibilite a q comme dans `train_dec3d_shared.py`,
les paves la remplacent.

ANCRAGE. La scene fixe est a l'etat z* = (s*, 0, 0) : s* = 0.818 (s normalise du decodeur),
estime en ajustant une courbe lisse p(s) aux ~200 positions triangulees de chacun des points
mobiles M10, M11, M12, M16 et en cherchant le s le plus proche des quatre points pointes dans
la scene fixe (ecart RMS 0.013). Chaque champ est ANCRE : sa sortie est f(z) - f(z*), donc a z*
le decodeur rend exactement les gaussiennes de base, au depart ET tout au long de
l'entrainement. Les images de la scene fixe supervisent ainsi directement la forme de base.

PORTEUR (zone q, `--carrier 1`). A l'initialisation, les gaussiennes de la zone q doivent
verifier dmu/dq ~ 0, dmu/ds = d((M12 + M10) / 2)/ds, dcov/dq ~ 0, dcov/ds ~ 0 : la languette
suit la pince en TRANSLATION, sans se deformer. On ajoute donc a leurs centres un terme FIXE
c(s) - c(s*), c = milieu des courbes p12(s), p10(s), polynomes de degre `--carrier-deg` en s
normalise ajustes sur les ~200 positions triangulees de chaque point (rig_manual.json, s lu a
l'instant affiche par click_points, `timebase.ref_index_of_shown`), s borne a la plage des
donnees. Le champ appris, nul au depart, est un RESIDU par-dessus. c(s*) - c(s*) = 0 : l'ancrage
a z* est conserve.
RACCORD : applique en tout-ou-rien a la zone q, le porteur cassait la scene au plan de M12M10
(face du haut de la zone q : la languette partait, la pince juste au-dessus restait). Chaque
gaussienne recoit donc le porteur avec un poids w : 1 dans la zone q, et dans la zone s
1 - smoothstep(d / bande), d = distance a la zone q, bande = `--carrier-band` h' ; 0 pour les
fixes. Licite : le porteur ne depend que de s, et la zone s a droit a s.

PORTE EN q (`--q-free-depth`, `--q-free-trans`, en h'). La partie de la pince qui est dans
la zone q (juste sous le plan de M12M10) ne doit pas dependre de q, et l'optimisation
l'apprenait mal (mouvement parasite en q). Coupure STRUCTURELLE : pour la zone q,
Delta = g Delta(s, q) + (1 - g) Delta(s, 0), g = 0 sur le plan et smoothstep jusqu'a 1 a
2/3 h' sous le plan (`--q-free-trans`), sans bande entierement figee (`--q-free-depth` 0) ;
plus bas g = 1, la languette est libre. Dans la bande, le mouvement en s appris
(contenu dans Delta(s, 0)) est garde, la dependance a q est exactement nulle ; a q = 0 les deux
evaluations coincident, donc l'ancrage a z* tient.

REPRISE (`--resume <dec.pt>`) : decodeur et corrections de couleur repris du checkpoint,
optimiseur neuf (montee du pas des champs comme au depart), compteur de pas a zero.

ANISOTROPIE (`--w-aniso`, demandee). relu(log(sigma_max / sigma_min) - log `--aniso-ratio`)^2,
sur les demi-axes de base de `--aniso-n` gaussiennes tirees par pas, et pour les mobiles de ce
tirage sur les demi-axes deformes a l'etat courant. Mesure au pas 36 000 de ft3 : 64 % des
gaussiennes au-dela du rapport 5, penalite moyenne 13.4. Elle penalise aussi les disques plats.

POIDS DE L'OBJET MOBILE (`--w-obj`, demande). Dans la L1 d'entrainement (4 vues et scene fixe),
les pixels pesent 1 + (`--w-obj` - 1) x couverture des gaussiennes de la zone q (fixees une fois
pour toutes), couverture obtenue par un second rendu gsplat de leur indicatrice (occultation
comprise, sans gradient) ; normalisation par la somme des poids. Tout sur GPU : le premier
masque (pave q projete, enveloppe convexe sur CPU pour chaque vue) divisait la vitesse par ~4.
Les evaluations restent NON ponderees.

PLANCHER DE COVARIANCE PAR LE PLUS PROCHE VOISIN (`--nn-floor 1`, demande). Au premier pas,
d_i = distance de la gaussienne i a sa plus proche voisine (positions de base, scene a z*),
calculee UNE FOIS et gardee dans le checkpoint (tampon `cov_floor`, non appris). La covariance
rendue est Sigma_i + d_i^2 I, donc sigma_min >= d_i : plus de gaussienne largement sous le
pixel. Terme isotrope, donc R diag(s^2) R^T + d^2 I = R diag(s^2 + d^2) R^T : il suffit de
rendre les demi-axes sqrt(s^2 + d^2) avec la meme rotation (fait dans `state`).
ECHEC (ft9, ft10) : meme borne, la scene fixe plafonne 1.1 dB sous ft8 ; abandonne au profit de
`--cov-eps` (epsilon UNIFORME eps^2 I, meme mecanisme, 0.0009 : 3.8 % des gaussiennes relevees).
`--nn-floor-max 3` borne d_i a 3 fois sa mediane : non borne (p99 17 fois la mediane), il
faisait de grosses taches floues dans les zones clairsemees (scene fixe plafonnee a 23.3 dB,
contre 25.8 avant le plancher).

PLANCHER DE TAILLE DE LA LANGUETTE (`--min-px-q`) : dans la zone q, les gaussiennes restees sous
ce nombre de pixels sur toutes les cameras sont agrandies jusqu'au seuil au lieu d'etre
supprimees, au demarrage et a chaque passage de l'elagage.

ELAGAGE DES GAUSSIENNES SOUS LE PIXEL (`--prune-small-px`, `--prune-small-every`). gsplat
elargit chaque gaussienne de 0.3 px^2 : une gaussienne bien plus petite qu'un pixel devient un
point d'environ un demi-pixel a pleine opacite, dont la couleur recoit tres peu de gradient
(mesure sur ft4 : 2 344 gaussiennes, 1.3 %, sous 0.5 px sur toutes les cameras d'entrainement,
dont 1 756 opaques ; points de mauvaise couleur visibles dans la demo). Sont supprimees, au
demarrage puis a chaque evaluation, les gaussiennes dont la plus grande taille a l'ecran
(plus grand demi-axe x f / z, position de base) reste sous le seuil sur toutes les cameras :
3 fixes, `--prune-small-n3` poses d'iphone3, `--prune-small-n2` vues de la scene fixe.
Parametres, lignes des champs, etats d'Adam, indices des zones, porte, porteur, graphe de
lissage et indicatrice de surponderation sont coupes aux memes indices.

TIRAGE PONDERE DES INSTANTS (`--w-q-edge`, `--w-s-far`, demande) : pour de bons resultats pres
du bord du domaine, un instant de la seance est tire avec une probabilite proportionnelle a
w_q x w_s, w_q = 1 + (`--w-q-edge` - 1) min(1, |q| / bord), |q| = norme 2 de (q0, q1) (unites
comparables ; bord = p99.5 de |q|), w_s = 1 + (`--w-s-far` - 1) |s - s*| / max |s - s*|. Meme effet en moyenne qu'une
ponderation de la perte, sans a-coups sur l'echelle du gradient.

GARDE-FOU (`--guard k`) : le meilleur checkpoint (L1 held-out de la seance) est garde dans
dec_best.pt ; si une evaluation est non finie ou depasse k fois la meilleure, il est recharge,
l'etat d'Adam est vide et la montee du pas des champs recommence. Instantanes tous les
`--snapshot-every` pas (dec_pas<N>.pt).

CONTRAINTES EXACTES (`--exact 1`, demande du 2026-09-30). Ce ne sont pas des initialisations
mais des contraintes de structure, vraies a tout instant de l'entrainement :
  - zone q (boite bleue) : mu(s, q) = mu_0 + c(s) - c(s*) + f(q), avec c(s) le milieu de M10M12,
    donc dmu/ds = dc/ds EXACTEMENT ; rotation, taille et couleur ne dependent que de q (le champ
    field_q est evalue a s = s*) ; delta des log-demi-axes = 0.5 (u(q) - u(0))^2 >= 0, donc la
    taille a q = 0 est la taille minimum et ne peut qu'augmenter avec q ; la rotation est libre ;
  - zone s seule (boite orange hors bleue) : centre et rotation dependent de s, taille non (delta
    des log-demi-axes nul) ; les tailles restent apprises, independamment de s.
La continuite spatiale de dmu/ds (zone s) et de dmu/dq (zone q) est celle de la perte de
coherence ci-dessous.

COHERENCE SPATIALE DU MOUVEMENT (`--w-smooth`, demandee). Deux gaussiennes proches doivent
bouger de facon proche, en moyenne ET en covariance, avec s ET avec q. Graphe des `--knn`
plus proches voisines de chaque gaussienne MOBILE (zone s ou q) parmi TOUTES les gaussiennes,
fixes comprises (les fixes de la frontiere), construit une fois sur la scene de la v3, arete
ponderee par exp(-d^2 / 2 sigma^2), sigma = longueur mediane. A l'etat courant z, derivees
par differences finies avant (pas `--smooth-h`) par rapport a s, q0, q1 du centre (porteur
compris), des log-demi-axes et de la rotation (axe-angle) : 27 nombres par gaussienne, nuls
pour une fixe. Perte : moyenne ponderee, sur `--smooth-edges` aretes tirees par pas, du carre
de l'ecart de ces derivees entre les deux extremites.

DONNEES, a chaque pas :
  - 1 instant de la seance multivue : 4 vues (iphone0, 1, 2 fixes + iphone3 pose), 256 x 256
    comme `train_dec3d_shared.py`, etat z(t) de l'encodeur gele (`z_all.npy`). Images
    RECTIFIEES avec la distorsion de `rig_manual.json` (convention cv2, verifiee : sur la
    nouvelle video, cette distorsion reduit le residu du PnP) : la scene statique est un
    stenope exact, sans rectification elle serait decalee vers les bords de ces vues ;
  - 1 image de la video de scene fixe (960 x 540, rectifiee), rendue a z*, poses raffinees et
    gains par image de la v3.
  Perte : L1 masquee sur les 4 vues + (1 - w) L1 + w (1 - SSIM) sur l'image fixe (la perte de
  son entrainement), + rappel des gains de la scene fixe vers l'identite.

COULEUR. Une affine 3 x 3 + biais par camera de la seance multivue (identite au depart) : elle
a ete filmee une semaine avant la scene fixe, sous un autre eclairage, et les couleurs des
gaussiennes viennent de la scene fixe. Gains et biais par image pour la scene fixe, repris de
la v3. `--color-q 0` (2026-10-07) : la couleur de la zone q ne depend plus de q ; elle part de
sa valeur a q = (0, 0) et reste apprise (sh0), sans dependance a q.

HELD-OUT : les `--val-sec` dernieres secondes de la seance multivue (comme `train_dec3d_shared`),
et les images held-out de la v3 pour la scene fixe (PSNR brut et apres gain/biais par canal).

Usage :
    py -u train_dec3d_zoned.py --out dec3d_zoned
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import mv_views as mv
import timebase

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))
from models_deform import SharedDeformField, _quat_mul, _axis_angle_quat   # noqa: E402

C0 = 0.28209479177387814


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', default='runs/static3dgs_v3/scene.pt')
    p.add_argument('--zones', default='runs/static_i3/zones.json')
    p.add_argument('--static-poses', default='runs/static_i3/poses_room.json')
    p.add_argument('--z-all', default='runs/dec3d_shared_p1/z_all.npy',
                   help='(s, q0, q1) normalises de l encodeur gele, une ligne par frame iphone1')
    p.add_argument('--s-star', type=float, default=0.818)
    p.add_argument('--rig', default='rig_manual.json')
    p.add_argument('--i3', default='i3_poses_dense_s1.json')
    p.add_argument('--side', type=int, default=256)
    p.add_argument('--undistort', type=int, default=1)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--max-dxyz', type=float, default=2.0)
    p.add_argument('--near', type=float, default=0.2)
    p.add_argument('--carrier', type=int, default=1)
    p.add_argument('--color-sigmoid', type=int, default=0,
                   help='1 : couleur des gaussiennes bornee par construction, sigmoid(logit), '
                        'deformation comprise (degre 0 seulement) ; conversion a la reprise')
    p.add_argument('--color-q', type=int, default=1,
                   help='0 : couleur de la zone q independante de q. A la reprise, la couleur '
                        'garde sa valeur a q = (0, 0) (sh0, l etat z* etant a q = 0) et les '
                        'lignes couleur du champ (s, q) sont mises et maintenues a zero ; sh0 '
                        'reste appris. 1 = couleur fonction de q (ancien)')
    p.add_argument('--native', type=int, default=0,
                   help='cote du carre des images NATIVES (make_mv_native.py, 960) ; 0 = caches '
                        'mv_<side>.npy')
    p.add_argument('--time-fix', type=int, default=1,
                   help='1 : conversions de temps corrigees (image iphone3 = frame posee, etat '
                        'simultane ; s des pointages lu a la frame iphone1 affichee) ; 0 = ancien')
    p.add_argument('--global-sim', type=int, default=0,
                   help='1 : similitude apprise repere de la seance -> scene fixe, appliquee aux '
                        'cameras fixes, a TOUTES les poses iphone3 et au porteur ; initialisee a la '
                        'reprise sur les corrections apprises des cameras fixes')
    p.add_argument('--lr-sim', type=float, default=1e-5)
    p.add_argument('--global-sim-i3', type=int, default=0,
                   help='1 : la similitude s applique aussi aux poses iphone3. MESURE (2026-10-01) : '
                        'a 0 par defaut, car ces poses sont localisees sur la carte map_i3, celle-la '
                        'meme qui a recale la scene fixe : leur appliquer la similitude degradait la '
                        'L1 held-out de l iphone3 de 0.024 a 0.055')
    p.add_argument('--learn-poses', type=int, default=0,
                   help='1 : corrections apprises des poses des cameras fixes (une par camera) '
                        'et de l iphone3 (spline lente en temps) ; scene fixe figee')
    p.add_argument('--lr-pose', type=float, default=2e-5)
    p.add_argument('--i3-exclude', default='',
                   help='.npy des etiquettes de temps des poses iphone3 a JETER (poses absurdes, '
                        'refine_worst.py)')
    p.add_argument('--i3-pose-corr', default='',
                   help='corrections PAR FRAME des poses iphone3 (refine_i3_poses.py), appliquees '
                        'aux instants d entrainement ; la spline lente repart alors de 0')
    p.add_argument('--i3-knot-sec', type=float, default=5.0)
    p.add_argument('--learn-s-fixed', type=int, default=0,
                   help='1 : s de la scene fixe appris (initialise a --s-star), q y reste 0')
    p.add_argument('--lr-s-fixed', type=float, default=2e-4)
    p.add_argument('--fast-v0', type=float, default=0.0,
                   help='poids de tirage des instants / (1 + (|ds/dt| / v0)^2), v0 en px d arc / s '
                        '(200) ; 0 = pas de poids de vitesse')
    p.add_argument('--fast-min', type=float, default=0.2, help='plancher du poids de vitesse')
    p.add_argument('--carrier-robust', type=int, default=1,
                   help='1 : ajustement robuste du porteur (pointages aberrants ecartes)')
    p.add_argument('--reset-q-below', type=float, default=0.0,
                   help='a la reprise, remet a zero la dependance a q (lignes du champ (s, q)) des '
                        'gaussiennes de la zone q dont la porte g est sous cette valeur ; 1.0 = '
                        'toute la bande de montee sous le plan de M12M10 (la pince)')
    p.add_argument('--refit-carrier', type=int, default=0,
                   help='1 : a la reprise, remplace le porteur du checkpoint par celui ajuste ici')
    p.add_argument('--carrier-deg', type=int, default=4)
    p.add_argument('--carrier-src', default='mid', choices=['mid', '12'],
                   help='mid : milieu des ajustements de M10 et M12 ; 12 : M12 seul decale du '
                        'vecteur median (M10 - M12) / 2')
    p.add_argument('--carrier-band', type=float, default=2.0,
                   help="largeur, en h', de la decroissance du porteur hors de la zone q")
    p.add_argument('--q-free-depth', type=float, default=0.0,
                   help="bande ENTIEREMENT sans dependance a q sous le plan de M12M10, en h'")
    p.add_argument('--q-free-trans', type=float, default=2.0 / 3.0,
                   help="largeur de la montee lisse g : 0 -> 1 sous cette bande, en h' "
                        "(0 et 0 : pas de porte)")
    p.add_argument('--q-gate-power', type=float, default=1.0,
                   help='g = smoothstep(t) ** p : p > 1 accentue la coupure (g reste plus longtemps '
                        'proche de 0), sans bande figee')
    p.add_argument('--resume', default='', help='checkpoint dec.pt a reprendre (fine-tuning)')
    p.add_argument('--exact', type=int, default=0,
                   help='1 : contraintes EXACTES (pas seulement a l init). Zone q : centre = base + '
                        'porteur c(s) - c(s*) + f(q), rotation, taille et couleur fonction de q '
                        'seul, taille jamais sous celle a q = 0 ; zone s : taille fixe en s')
    p.add_argument('--sh-degree', type=int, default=-1,
                   help='tronque les harmoniques spheriques a ce degre (0 = couleur seule, '
                        'sans dependance a la direction) ; -1 = garder celui du checkpoint')
    p.add_argument('--w-obj', type=float, default=1.0,
                   help='poids de reconstruction des pixels du pave q projete (1 = aucun)')
    p.add_argument('--w-aniso', type=float, default=0.0)
    p.add_argument('--aniso-ratio', type=float, default=5.0)
    p.add_argument('--aniso-n', type=int, default=32768)
    p.add_argument('--w-big', type=float, default=0.0,
                   help='poids de la penalite de taille : relu(log demi-axe max - log(--big-ratio x '
                        'mediane des demi-axes max))^2, sur --aniso-n gaussiennes tirees (taille '
                        'deformee comprise pour les mobiles) ; 0 = aucune')
    p.add_argument('--big-ratio', type=float, default=10.0)
    p.add_argument('--max-gauss', type=int, default=0,
                   help='nombre maximum de gaussiennes : a chaque elagage, s il est depasse, les '
                        'gaussiennes FIXES de plus faible contribution (taille ecran x opacite) sont '
                        'supprimees ; 0 = pas de plafond')
    p.add_argument('--prune-small-px', type=float, default=0.0,
                   help='supprime les gaussiennes plus petites que ce nombre de pixels sur toutes '
                        'les cameras d entrainement (0 = jamais)')
    p.add_argument('--prune-small-every', type=int, default=2000)
    p.add_argument('--nn-floor', type=int, default=0,
                   help='1 : plancher de covariance d_i^2 I, d_i = distance au plus proche voisin, '
                        'calcule une fois (repris du checkpoint s il y est deja)')
    p.add_argument('--cov-eps', type=float, default=0.0,
                   help='epsilon diagonal UNIFORME ajoute a toutes les covariances (eps^2 I, meme '
                        'mecanisme que --nn-floor) ; 0.0009 garantit 0.5 px a toutes les '
                        'gaussiennes sur leur meilleure camera d entrainement (mesure sur ft8)')
    p.add_argument('--nn-floor-max', type=float, default=0.0,
                   help='borne d_i a ce multiple de sa mediane (0 = pas de borne) ; appliquee aussi '
                        'a un plancher repris du checkpoint')
    p.add_argument('--min-px-q', type=float, default=0.0,
                   help='zone q (languette) : au lieu d etre supprimees, les gaussiennes dont la plus '
                        'grande taille a l ecran reste sous ce nombre de pixels sur toutes les '
                        'cameras sont AGRANDIES (trois demi-axes, meme facteur) jusqu a ce seuil ; '
                        '0 = pas de plancher (elles suivent --prune-small-px)')
    p.add_argument('--prune-small-n3', type=int, default=600)
    p.add_argument('--prune-small-n2', type=int, default=300)
    p.add_argument('--w-q-edge', type=float, default=1.0,
                   help='poids de tirage au bord du domaine en q (1 = uniforme)')
    p.add_argument('--w-s-far', type=float, default=1.0,
                   help='poids de tirage pour le s le plus loin de s* (1 = uniforme)')
    p.add_argument('--guard', type=float, default=0.0,
                   help='recharge dec_best.pt si la L1 held-out depasse ce multiple de la meilleure '
                        '(0 = pas de garde-fou)')
    p.add_argument('--snapshot-every', type=int, default=0)
    p.add_argument('--w-smooth', type=float, default=1.0)
    p.add_argument('--knn', type=int, default=6)
    p.add_argument('--smooth-edges', type=int, default=4096)
    p.add_argument('--smooth-h', type=float, default=0.05)
    p.add_argument('--steps', type=int, default=100000)
    p.add_argument('--lr-xyz', type=float, default=2e-5,
                   help='centres : valeur de fin de 3DGS a l echelle de la scene (1.6e-6 x 11)')
    p.add_argument('--lr-scale', type=float, default=5e-3)
    p.add_argument('--lr-quat', type=float, default=1e-3)
    p.add_argument('--lr-alpha', type=float, default=5e-2)
    p.add_argument('--lr-sh0', type=float, default=2.5e-3)
    p.add_argument('--lr-shN', type=float, default=1.25e-4)
    p.add_argument('--lr-field', type=float, default=5e-4)
    p.add_argument('--warmup', type=int, default=2000)
    p.add_argument('--lr-color', type=float, default=1e-3)
    p.add_argument('--color-reg', type=float, default=1e-2)
    p.add_argument('--ssim-w', type=float, default=0.2)
    p.add_argument('--w-static', type=float, default=1.0,
                   help='poids de la perte de la scene fixe face a celle des 4 vues')
    p.add_argument('--exclude-poses', default='runs/static_i3/excluded_poses.json',
                   help='images de la scene fixe ecartees (poses absurdes, static_points_tool.py)')
    p.add_argument('--i0-max-px', type=float, default=2.0)
    p.add_argument('--val-sec', type=float, default=60.0)
    p.add_argument('--eval-every', type=int, default=2000)
    p.add_argument('--print-every', type=int, default=100)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='dec3d_zoned')
    return p.parse_args()


def ssim(x, y, win=11, sigma=1.5):
    g = torch.exp(-(torch.arange(win, device=x.device) - win // 2) ** 2 / (2 * sigma ** 2))
    g = (g / g.sum()).float()
    k = (g[:, None] * g[None, :])[None, None].repeat(3, 1, 1, 1)
    mu_x, mu_y = F.conv2d(x, k, groups=3), F.conv2d(y, k, groups=3)
    sxx = F.conv2d(x * x, k, groups=3) - mu_x ** 2
    syy = F.conv2d(y * y, k, groups=3) - mu_y ** 2
    sxy = F.conv2d(x * y, k, groups=3) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mu_x * mu_y + c1) * (2 * sxy + c2))
            / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))).mean()


def inside(X, b):
    R = torch.tensor(b['R'], dtype=X.dtype, device=X.device)
    loc = (X - torch.tensor(b['center'], dtype=X.dtype, device=X.device)) @ R
    return (loc.abs() <= torch.tensor(b['half'], dtype=X.dtype, device=X.device)).all(1)


class ZonedSplat(nn.Module):
    """Gaussiennes de base (toutes optimisees) + deux champs ancres a z* : `field_s` (entree
    s) pour `idx_s`, `field_q` (entree s, q0, q1) pour `idx_q`."""

    def __init__(self, sc, idx_s, idx_q, z_star, hidden, layers, max_dxyz,
                 carrier=None, carrier_rng=None, carrier_w=None):
        super().__init__()
        # poids du porteur par gaussienne ; par defaut 1 sur la zone q, 0 ailleurs
        if carrier_w is None:
            carrier_w = torch.zeros(sc['xyz'].shape[0])
            carrier_w[idx_q.cpu()] = 1.0
        self.register_buffer('carrier_w', torch.as_tensor(carrier_w, dtype=torch.float32))
        # porteur de la zone q : coefficients (3, deg+1) de c(s), puissances decroissantes
        # (np.polyfit), et plage de s des donnees ; nul si absent
        self.register_buffer('carrier_coef', torch.zeros(3, 1) if carrier is None
                             else torch.as_tensor(carrier, dtype=torch.float32))
        self.register_buffer('carrier_rng', torch.tensor([-1e9, 1e9]) if carrier_rng is None
                             else torch.as_tensor(carrier_rng, dtype=torch.float32))
        self.xyz = nn.Parameter(sc['xyz'].float().clone())
        self.quat = nn.Parameter(F.normalize(sc['quat'].float(), dim=-1).clone())
        self.log_scale = nn.Parameter(torch.log(sc['scale'].float().clamp(min=1e-6)))
        self.logit_alpha = nn.Parameter(torch.logit(sc['alpha'].float().clamp(1e-4, 1 - 1e-4)))
        self.sh0 = nn.Parameter(sc['sh'][:, :1].float().clone())
        self.shN = nn.Parameter(sc['sh'][:, 1:].float().clone())
        self.sh_degree = int(sc['sh_degree'])
        self.register_buffer('idx_s', idx_s)
        self.register_buffer('idx_q', idx_q)
        # plancher de covariance (d_i, distance au plus proche voisin) ; 0 = aucun
        self.register_buffer('cov_floor', torch.zeros(sc['xyz'].shape[0]))
        # porte de la dependance a q, par gaussienne de la zone q (1 = libre, 0 = aucune)
        self.register_buffer('q_gate', torch.ones(len(idx_q)))
        # contraintes exactes (`--exact`) : 1 = actives ; tampon pour que demo et export les
        # retrouvent dans le checkpoint
        self.register_buffer('exact', torch.zeros(()))
        # couleur bornee (`--color-sigmoid`) : 1 = sh0 contient le LOGIT de la couleur, rendue
        # sigmoid(u) dans (0, 1) quelle que soit la deformation ; 0 = sh0 = harmonique d'ordre 0
        self.register_buffer('color_sigmoid', torch.zeros(()))
        # `--color-q 0` : la couleur de la zone q ne depend pas de q (delta de couleur de
        # field_q ignore ; ses lignes de sortie sont en plus tenues a zero par l'entrainement,
        # pour que demo et export, qui ne connaissent pas ce drapeau, donnent la meme chose)
        self.color_q = True
        # similitude APPRISE repere de la seance (rig, pointages) -> repere de la scene fixe
        # (`--global-sim`) : (rotation axe-angle (3), log echelle (1), translation (3)) ;
        # x_fixe = e^l R x_seance + t. Nulle = identite. Le porteur, triangule dans le repere de
        # la seance, est tourne et mis a l'echelle par elle.
        self.world_sim = nn.Parameter(torch.zeros(7))
        self.register_buffer('z_star', z_star)
        self.max_dxyz = max_dxyz
        self.field_s = SharedDeformField(1, len(idx_s), 12, hidden=hidden, layers=layers)
        self.field_q = SharedDeformField(3, len(idx_q), 12, hidden=hidden, layers=layers)

    def _delta(self, field, cond, cond0, rows=None, gate=None, kind='s'):
        def split(r):
            return (torch.tanh(r[..., 0:3]) * self.max_dxyz, torch.tanh(r[..., 3:6]),
                    torch.tanh(r[..., 6:9]) * math.pi, torch.tanh(r[..., 9:12]) * 0.5)
        exact = bool(self.exact > 0)
        if exact and kind == 'q':
            # zone q : aucune dependance a s hors du porteur (champ evalue a s*)
            cond = cond.clone()
            cond[:, 0] = cond0[:, 0]
        ref = split(field(cond0, rows))
        full = [u - v for u, v in zip(split(field(cond, rows)), ref)]
        if exact:
            if kind == 's':
                full[1] = torch.zeros_like(full[1])        # zone s : taille fixe en s
            else:
                full[1] = 0.5 * full[1] ** 2               # zone q : taille >= celle a q = 0
        if gate is None or bool((gate == 1).all()):
            return full
        c0 = cond.clone()
        c0[:, 1:] = 0.0                                   # meme s, q = 0
        noq = [u - v for u, v in zip(split(field(c0, rows)), ref)]
        g = gate[None, :, None]
        return [g * u + (1 - g) * v for u, v in zip(full, noq)]

    def motion(self, z, gidx):
        """Deformation (B, M, 9) des gaussiennes d'indices GLOBAUX `gidx` a l'etat z :
        deplacement du centre (porteur compris), des log-demi-axes, rotation axe-angle.
        Nulle pour une gaussienne fixe. Seules les lignes utiles des champs sont evaluees."""
        if not hasattr(self, '_pos'):
            pos = torch.full((2, self.xyz.shape[0]), -1, dtype=torch.long, device=z.device)
            pos[0, self.idx_s] = torch.arange(len(self.idx_s), device=z.device)
            pos[1, self.idx_q] = torch.arange(len(self.idx_q), device=z.device)
            self._pos = pos
        B, M = z.shape[0], len(gidx)
        out = torch.zeros(B, M, 9, device=z.device)
        zs = self.z_star.unsqueeze(0)
        for k, (field, cond, cond0) in enumerate(((self.field_s, z[:, :1], zs[:, :1]),
                                                  (self.field_q, z, zs))):
            loc = self._pos[k, gidx]
            sel = torch.nonzero(loc >= 0, as_tuple=True)[0]
            if len(sel) == 0:
                continue
            dx, dl, dr, _ = self._delta(field, cond, cond0, loc[sel],
                                        None if k == 0 else self.q_gate[loc[sel]],
                                        'sq'[k])
            out[:, sel] = torch.cat([dx, dl, dr], -1)
        tr = self.carrier(z[:, 0]) - self.carrier(self.z_star[None, 0])
        out[..., :3] = out[..., :3] + self.carrier_w[gidx][None, :, None] * tr[:, None, :]
        return out

    def sim_rs(self):
        """(R (3, 3), echelle) de la similitude seance -> scene fixe."""
        w = self.world_sim[:3]
        th = w.norm().clamp(min=1e-12)
        k = w / th
        z = torch.zeros((), device=w.device)
        Kx = torch.stack([torch.stack([z, -k[2], k[1]]), torch.stack([k[2], z, -k[0]]),
                          torch.stack([-k[1], k[0], z])])
        R = torch.eye(3, device=w.device) + torch.sin(th) * Kx + (1 - torch.cos(th)) * Kx @ Kx
        return R, torch.exp(self.world_sim[3])

    def carrier(self, s):
        """c(s) (B, 3), s (B,) borne a la plage des donnees ; Horner, puis similitude seance ->
        scene fixe (la translation est omise : le porteur n'entre que par des differences)."""
        s = s.clamp(self.carrier_rng[0], self.carrier_rng[1])
        out = torch.zeros(s.shape[0], 3, device=s.device)
        for k in range(self.carrier_coef.shape[1]):
            out = out * s[:, None] + self.carrier_coef[:, k]
        R, sc = self.sim_rs()
        return sc * out @ R.T

    def state(self, z):
        """z (B, 3) = (s, q0, q1) normalises -> xyz, quat, log_scale (B, K, .), sh0 (B, K, 1, 3)."""
        B = z.shape[0]
        xyz = self.xyz.unsqueeze(0).repeat(B, 1, 1)
        quat = F.normalize(self.quat, dim=-1).unsqueeze(0).repeat(B, 1, 1)
        logs = self.log_scale.unsqueeze(0).repeat(B, 1, 1)
        sh0 = self.sh0.unsqueeze(0).repeat(B, 1, 1, 1)
        zs = self.z_star.unsqueeze(0)
        for field, idx, cond, cond0, gate in ((self.field_s, self.idx_s, z[:, :1], zs[:, :1],
                                               None),
                                              (self.field_q, self.idx_q, z, zs, self.q_gate)):
            if len(idx) == 0:
                continue
            dx, dl, dr, dc = self._delta(field, cond, cond0, None, gate,
                                         's' if field is self.field_s else 'q')
            xyz[:, idx] = xyz[:, idx] + dx
            logs[:, idx] = logs[:, idx] + dl
            quat[:, idx] = F.normalize(_quat_mul(_axis_angle_quat(dr), quat[:, idx]), dim=-1)
            if field is self.field_q and not self.color_q:
                continue
            # delta de couleur : en unites de couleur (au plus +-0.5), ramene a l'harmonique ;
            # en logit (pente de la sigmoide 1/4 en 0.5, d'ou le facteur 4) si la couleur est bornee
            sh0[:, idx] = sh0[:, idx] + ((4.0 * dc) if bool(self.color_sigmoid > 0)
                                         else (dc / C0)).unsqueeze(2)
        tr = self.carrier(z[:, 0]) - self.carrier(self.z_star[None, 0])
        xyz = xyz + self.carrier_w[None, :, None] * tr[:, None, :]
        if bool(self.color_sigmoid > 0):
            # gsplat rend 0.5 + C0 sh0 au degre 0 : on lui passe l'harmonique de sigmoid(u)
            sh0 = (torch.sigmoid(sh0) - 0.5) / C0
        if bool((self.cov_floor > 0).any()):
            # Sigma + d^2 I = R diag(s^2 + d^2) R^T : demi-axes effectifs sqrt(s^2 + d^2)
            logs = 0.5 * torch.log(torch.exp(2 * logs) + self.cov_floor[None, :, None] ** 2)
        return xyz, quat, logs, sh0

    def render(self, z, viewmats, Ks, W, H, near=0.2, ind=None):
        """z (B, 3), viewmats (B, V, 4, 4), Ks (B, V, 3, 3) -> (B, V, 3, H, W). Avec `ind`
        (K,) : renvoie aussi la COUVERTURE (B, V, 1, H, W) de ces gaussiennes (indicatrice
        compositee, occultation comprise), sans gradient, par un second rendu a un canal."""
        from gsplat import rasterization
        xyz, quat, logs, sh0 = self.state(z)
        al = torch.sigmoid(self.logit_alpha).clamp(0, 0.99)
        out, cov = [], []
        for b in range(z.shape[0]):
            r, _, _ = rasterization(
                means=xyz[b], quats=quat[b], scales=torch.exp(logs[b]), opacities=al,
                colors=torch.cat([sh0[b], self.shN], 1), viewmats=viewmats[b], Ks=Ks[b],
                width=W, height=H, near_plane=near, far_plane=100.0,
                sh_degree=self.sh_degree, render_mode='RGB')
            out.append(r.permute(0, 3, 1, 2).clamp(0, 1))
            if ind is not None:
                with torch.no_grad():
                    c_, _, _ = rasterization(
                        means=xyz[b].detach(), quats=quat[b].detach(),
                        scales=torch.exp(logs[b]).detach(), opacities=al.detach(),
                        colors=ind[:, None], viewmats=viewmats[b], Ks=Ks[b], width=W, height=H,
                        near_plane=near, far_plane=100.0, sh_degree=None, render_mode='RGB')
                cov.append(c_.permute(0, 3, 1, 2).clamp(0, 1))
        if ind is None:
            return torch.stack(out, 0)
        return torch.stack(out, 0), torch.stack(cov, 0)


def scene_from_ckpt(d, sh_degree):
    """Geometrie de base au format de la scene statique, lue dans l'etat `d` d'un checkpoint
    ZonedSplat : sert a reconstruire le decodeur a la taille DU CHECKPOINT (qui peut differer de
    la v3 apres duplicate_zoned.py)."""
    # degre lu sur la forme des coefficients (un checkpoint tronque par --sh-degree en a moins)
    deg = min(int(sh_degree), int(round(math.sqrt(d['shN'].shape[1] + 1))) - 1)
    return dict(xyz=d['xyz'], quat=d['quat'], scale=torch.exp(d['log_scale']),
                alpha=torch.sigmoid(d['logit_alpha']), sh=torch.cat([d['sh0'], d['shN']], 1),
                sh_degree=deg)


def shown_proxy(cam, t):
    """Indice de frame du proxy de `cam` que `click_points.py` a AFFICHE a l'etiquette `t`."""
    return (round((float(t) + timebase.OFFSET[cam]) * timebase.FPS_LABEL)
            - timebase.PROXY_START[cam])


def proxy_to_ref(cam, j):
    """Indice (fractionnaire) du proxy de la camera de reference (mv.REF, qui indexe z(t))
    simultane a la frame `j` du proxy de `cam`."""
    i0 = (float(j) - timebase.B[cam]) / (1.0 + timebase.A[cam])
    return i0 * (1.0 + timebase.A[mv.REF]) + timebase.B[mv.REF]


class JpegBlob:
    """Frames JPEG d'une camera ecrites par `make_mv_native.py` : blob + debuts ; frames[j] donne
    l'image (3, C, C) uint8 RGB, rectifiee et posee au centre du carre."""

    def __init__(self, cam, C):
        self.buf = np.memmap(HERE / cam / ('mvn_%d.bin' % C), dtype=np.uint8, mode='r')
        self.off = np.load(HERE / cam / ('mvn_%d_off.npy' % C))
        self.meta = json.loads((HERE / cam / ('mvn_%d_meta.json' % C)).read_text(encoding='utf-8'))
        self.mask = cv2.imread(str(HERE / cam / 'mvn_mask.png'), cv2.IMREAD_GRAYSCALE) > 0

    def __len__(self):
        return len(self.off) - 1

    def __getitem__(self, j):
        im = cv2.imdecode(np.asarray(self.buf[self.off[j]:self.off[j + 1]]), cv2.IMREAD_COLOR)
        return im[:, :, ::-1].transpose(2, 0, 1)


def fit_carrier(a):
    """Coefficients (3, deg+1) de c(s) = (p12(s) + p10(s)) / 2 et plage de s des donnees."""
    r = json.loads((HERE / a.rig).read_text(encoding='utf-8'))['points3d']
    Z = np.load(HERE / a.z_all)
    fits, lo, hi, kept = [], [], [], {}
    for k in ('12', '10'):
        rows = []
        for kk, v in r.items():
            if kk.startswith('M_') and kk.endswith('_' + k):
                # CORRIGE (2026-09-30) : `ref_index_of_shown` renvoie un indice de l'iphone0,
                # alors que z(t) est indexe sur l'iphone1 (mv.REF) : jusqu'a 8 frames d'ecart
                j = (shown_proxy(mv.REF, kk.split('_')[1]) if a.time_fix else
                     int(round(timebase.ref_index_of_shown(mv.REF, kk.split('_')[1]))))
                if 0 <= j < len(Z):
                    rows.append((Z[j, 0], v, kk.split('_')[1]))
        sv = np.array([x[0] for x in rows])
        X = np.array([x[1] for x in rows])
        keep = np.ones(len(sv), bool)
        for _ in range(10 if a.carrier_robust else 0):
            # ajustement robuste : un pointage aberrant (M10 et M12 intervertis, pointage pendant
            # un mouvement rapide) tirait l'extremite du polynome ; on ecarte les residus au-dela
            # de mediane + 3 x MAD et on reajuste jusqu'a stabilite
            C_ = np.stack([np.polyfit(sv[keep], X[keep, d], a.carrier_deg) for d in range(3)])
            e = np.linalg.norm(X - np.stack([np.polyval(C_[d], sv) for d in range(3)], 1), axis=1)
            ek = e[keep]
            new = e < np.median(ek) + 3 * 1.4826 * np.median(np.abs(ek - np.median(ek)))
            if (new == keep).all():
                break
            keep = new
        print('porteur M%s : %d pointages, %d ecartes par l ajustement robuste'
              % (k, len(sv), int((~keep).sum())))
        fits.append(np.stack([np.polyfit(sv[keep], X[keep, d], a.carrier_deg)
                              for d in range(3)]))
        kept[k] = {rows[i][2]: X[i] for i in np.flatnonzero(keep)}
        lo.append(sv.min())
        hi.append(sv.max())
    if a.carrier_src == '12':
        # M12 seul (le point le mieux pointe : erreur leave-one-out 3.6 a 4.6 mm, contre 12 a
        # 23 mm pour M10), decale d'un vecteur CONSTANT pour retrouver le milieu de M10M12 :
        # mediane de (M10 - M12) / 2 aux instants ou les deux pointages sont gardes
        both = sorted(set(kept['12']) & set(kept['10']))
        off = np.median(np.stack([0.5 * (kept['10'][t] - kept['12'][t]) for t in both]), 0)
        print('porteur : M12 seul, degre %d, decale de (M10 - M12) / 2 = %s (mediane sur %d '
              'instants)' % (a.carrier_deg, np.round(off, 4).tolist(), len(both)))
        C = fits[0].copy()
        C[:, -1] += off
        return C, [float(lo[0]), float(hi[0])]
    return 0.5 * (fits[0] + fits[1]), [float(max(lo)), float(min(hi))]


def undistort_grid(K, dist, W, H, dev):
    """Grille grid_sample : pour chaque pixel de l'image RECTIFIEE (meme K), sa source dans
    l'image distordue ; et le masque des pixels dont la source est dans l'image."""
    d5 = np.zeros(5)
    d5[:len(dist)] = dist
    mx, my = cv2.initUndistortRectifyMap(np.asarray(K, float), d5, None, np.asarray(K, float),
                                         (W, H), cv2.CV_32FC1)
    ok = (mx >= 0) & (mx <= W - 1) & (my >= 0) & (my <= H - 1)
    g = np.stack([mx / (W - 1) * 2 - 1, my / (H - 1) * 2 - 1], -1)
    return (torch.tensor(g, dtype=torch.float32, device=dev)[None],
            torch.tensor(ok, dtype=torch.float32, device=dev))


def main():
    a = parse()
    dev = 'cuda'
    torch.manual_seed(a.seed)
    run = HERE / 'runs' / a.out
    run.mkdir(parents=True, exist_ok=True)
    logf = open(run / 'log.txt', 'a', encoding='utf-8')

    def say(s):
        print(s, flush=True)
        logf.write(s + '\n')
        logf.flush()

    # ── scene statique et zones ─────────────────────────────────────────────────
    sc = torch.load(HERE / a.scene, map_location=dev, weights_only=False)
    zones = json.loads((HERE / a.zones).read_text(encoding='utf-8'))
    ck_res = None
    if a.resume:
        # reprise : geometrie et zones DU CHECKPOINT (le nombre de gaussiennes a pu changer,
        # duplicate_zoned.py) ; la v3 ne sert plus qu'aux donnees de la scene fixe
        ck_res = torch.load(HERE / a.resume, map_location=dev, weights_only=False)
        d_ = ck_res['dec']
        scg = scene_from_ckpt(d_, sc['sh_degree'])
        idx_s, idx_q = d_['idx_s'].long(), d_['idx_q'].long()
        X = d_['xyz'].float()
        n_fix = len(X) - len(idx_s) - len(idx_q)
        say('reprise de %s (pas %d) : %d gaussiennes ; zone q %d, zone s seule %d, fixes %d'
            % (a.resume, ck_res['step'], len(X), len(idx_q), len(idx_s), n_fix))
    else:
        scg = sc
        X = sc['xyz'].float()
        in_s, in_q = inside(X, zones['zone_s']), inside(X, zones['zone_q'])
        idx_q = torch.nonzero(in_q, as_tuple=True)[0]
        idx_s = torch.nonzero(in_s & ~in_q, as_tuple=True)[0]
        n_fix = int((~in_s & ~in_q).sum())
        say('%s : %d gaussiennes ; zone q (s et q) %d, zone s seule (s) %d, fixes %d ; %d de la '
            'zone q hors de la zone s' % (a.scene, len(X), len(idx_q), len(idx_s), n_fix,
                                         int((in_q & ~in_s).sum())))
    z_star = torch.tensor([a.s_star, 0.0, 0.0], device=dev)
    carrier, crng = None, None
    if a.carrier:
        carrier, crng = fit_carrier(a)
        say('porteur zone q : c(s) = milieu de p12(s), p10(s), degre %d, s dans [%.2f, %.2f]'
            % (a.carrier_deg, crng[0], crng[1]))
    cw = None
    if a.carrier:
        # distance a la zone q, dans son repere : norme de l'exces sur les demi-cotes
        bq = zones['zone_q']
        Rq = torch.tensor(bq['R'], device=dev)
        loc = (X - torch.tensor(bq['center'], device=dev)) @ Rq
        dq = (loc.abs() - torch.tensor(bq['half'], device=dev)).clamp(min=0).norm(dim=1)
        band = a.carrier_band * float(bq['h_prime'])
        t_ = (dq / band).clamp(0, 1)
        cw = torch.zeros(len(X), device=dev)
        cw[idx_s] = (1 - (3 * t_ ** 2 - 2 * t_ ** 3))[idx_s]
        cw[idx_q] = 1.0
        say('raccord du porteur : bande %.3f ; zone s : %d gaussiennes a w > 0.5, %d a 0 < w <= '
            '0.5, %d a w = 0' % (band, int((cw[idx_s] > 0.5).sum()),
                                 int(((cw[idx_s] > 0) & (cw[idx_s] <= 0.5)).sum()),
                                 int((cw[idx_s] == 0).sum())))
    dec = ZonedSplat(scg, idx_s, idx_q, z_star, a.hidden, a.layers, a.max_dxyz,
                     carrier, crng, cw).to(dev)
    if a.carrier:
        with torch.no_grad():
            for sv in (-1.5, 0.0, 1.2):
                zz = torch.tensor([[sv, 0.0, 0.0]], device=dev)
                say('  s = %+.1f : translation de la zone q %s'
                    % (sv, np.round((dec.carrier(zz[:, 0]) - dec.carrier(z_star[None, 0]))
                                    [0].cpu().numpy(), 3).tolist()))
    if a.q_free_depth > 0 or a.q_free_trans > 0:
        bq = zones['zone_q']
        Rq = torch.tensor(bq['R'], device=dev)
        loc = (X[idx_q] - torch.tensor(bq['center'], device=dev)) @ Rq
        hp_ = float(bq['h_prime'])
        depth = float(bq['half'][2]) - loc[:, 2]           # sous la face du haut (plan M12M10)
        t_ = ((depth - a.q_free_depth * hp_) / max(a.q_free_trans * hp_, 1e-9)).clamp(0, 1)
        with torch.no_grad():
            dec.q_gate.copy_((3 * t_ ** 2 - 2 * t_ ** 3) ** a.q_gate_power)
        say("porte en q : bande figee %.3f sous le plan de M12M10, montee sur %.3f, puissance %.1f ;"
            " zone q : %d a g = 0, %d en transition, %d a g = 1"
            % (a.q_free_depth * hp_, a.q_free_trans * hp_, a.q_gate_power,
               int((dec.q_gate == 0).sum()),
               int(((dec.q_gate > 0) & (dec.q_gate < 1)).sum()), int((dec.q_gate == 1).sum())))
    if ck_res is not None:
        drop_ = {'q_gate'} | ({'carrier_coef', 'carrier_rng'} if a.refit_carrier else set())
        sd = {k: v for k, v in ck_res['dec'].items() if k not in drop_}
        miss = dec.load_state_dict(sd, strict=False)
        say('  etat repris ; tampons absents gardes a leur valeur : %s'
            % [k for k in miss.missing_keys])
        if a.reset_q_below > 0:
            sel_ = torch.nonzero(dec.q_gate < a.reset_q_below, as_tuple=True)[0]
            rows_ = (sel_[:, None] * 12 + torch.arange(12, device=dev)[None]).reshape(-1)
            with torch.no_grad():
                dec.field_q.head.weight[rows_] = 0.0
                dec.field_q.head.bias[rows_] = 0.0
            say('dependance a q remise a zero pour %d gaussiennes de la zone q (porte < %.2f)'
                % (len(sel_), a.reset_q_below))
        if a.refit_carrier and carrier is not None:
            # le decodeur a ete construit avec le porteur ajuste ici (degre eventuellement
            # different) ; on le compare a celui du checkpoint, relatif a s*
            co = ck_res['dec']['carrier_coef'].cpu().numpy()
            ss_ = np.linspace(float(crng[0]), float(crng[1]), 200)
            ev = lambda C_, x_: np.stack([np.polyval(C_[d], x_) for d in range(3)], -1)
            dd = (ev(np.asarray(carrier), ss_) - ev(np.asarray(carrier), a.s_star)) - \
                 (ev(co, ss_) - ev(co, a.s_star))
            say('porteur remplace (%s, degre %d) : ecart au porteur du checkpoint, relatif a s*, '
                'median %.4f, max %.4f' % ('M12 seul' if a.carrier_src == '12' else 'milieu',
                                          a.carrier_deg, float(np.median(np.linalg.norm(dd, axis=1))),
                                          float(np.linalg.norm(dd, axis=1).max())))
    def zero_color_q():
        """Lignes couleur (9:12) du champ (s, q), pour chaque gaussienne de la zone q, a zero."""
        rows_ = (torch.arange(dec.field_q.n_gaussians, device=dev)[:, None] * 12
                 + torch.arange(9, 12, device=dev)[None]).reshape(-1)
        with torch.no_grad():
            dec.field_q.head.weight[rows_] = 0.0
            dec.field_q.head.bias[rows_] = 0.0

    if not a.color_q:
        # couleur a q = (0, 0) : z* = (s*, 0, 0) et le champ est ancre a z*, donc a q = 0 le
        # delta de couleur est nul (sans --exact, il l'est a s = s*) et la couleur vaut sh0 ;
        # retirer la dependance garde donc exactement cette couleur
        if not a.exact:
            say('ATTENTION : --color-q 0 sans --exact, la couleur gardee est celle a z* = (s*, 0, 0)')
        with torch.no_grad():
            zq_ = torch.tensor([[a.s_star, 1.0, 1.0]], device=dev)
            _, _, _, c1_ = dec.state(zq_)
            dec.color_q = False
            _, _, _, c0_ = dec.state(zq_)
            dc_ = (c1_ - c0_)[0, dec.idx_q].abs().max()
        zero_color_q()
        dec.color_q = False
        say('couleur independante de q : couleur de la zone q a q = (0, 0) gardee (sh0, appris), '
            'lignes couleur du champ (s, q) a zero (%d gaussiennes) ; ecart retire a q = (1, 1) : '
            'max %.4f en harmonique' % (dec.field_q.n_gaussians, float(dc_)))
    if a.exact:
        with torch.no_grad():
            dec.exact.fill_(1.0)
        say('contraintes exactes : zone q translatee par le porteur seul en s (champ de q seul, '
            'taille >= taille a q = 0), zone s a taille fixe en s')
    if 0 <= a.sh_degree < dec.sh_degree:
        # la moyenne sur les directions des harmoniques de degre >= 1 est nulle : sh0 porte deja
        # la couleur moyenne, on retire seulement la dependance a la direction
        n_ = (a.sh_degree + 1) ** 2 - 1
        say('harmoniques spheriques tronquees : degre %d -> %d (%d coefficients par couleur)'
            % (dec.sh_degree, a.sh_degree, n_ + 1))
        dec.shN = nn.Parameter(dec.shN.detach()[:, :n_].clone())
        dec.sh_degree = a.sh_degree
    if a.color_sigmoid and not bool(dec.color_sigmoid > 0):
        with torch.no_grad():
            c_ = 0.5 + C0 * dec.sh0.data
            n_out = int(((c_ > 0.99) | (c_ < 0.01)).any(-1).sum())
            dec.sh0.data.copy_(torch.logit(c_.clamp(0.01, 0.99)))
            dec.color_sigmoid.fill_(1.0)
        say('couleur bornee par une sigmoide : %d gaussiennes avaient une couleur hors de '
            '(0.01, 0.99), ramenee dans cet intervalle' % n_out)
    if a.cov_eps > 0:
        with torch.no_grad():
            dec.cov_floor.fill_(a.cov_eps)
        sm_ = torch.exp(dec.log_scale.detach()).min(1).values
        say('epsilon diagonal uniforme %.5f : %.1f %% des gaussiennes ont un sigma_min en dessous'
            % (a.cov_eps, 100 * float((sm_ < a.cov_eps).float().mean())))
    if a.nn_floor:
        if bool((dec.cov_floor > 0).any()):
            say('plancher de covariance repris du checkpoint : d median %.4f'
                % float(dec.cov_floor.median()))
        else:
            from scipy.spatial import cKDTree
            Xb = dec.xyz.detach().cpu().numpy()
            dd, _ = cKDTree(Xb).query(Xb, k=2)
            with torch.no_grad():
                dec.cov_floor.copy_(torch.tensor(dd[:, 1], dtype=torch.float32, device=dev))
            f_ = dec.cov_floor
            sm_ = torch.exp(dec.log_scale.detach()).min(1).values
            say('plancher de covariance d_i (plus proche voisin, calcule une fois) : median %.4f, '
                'p99 %.4f ; %.1f %% des gaussiennes ont un sigma_min < d_i (releve)'
                % (float(f_.median()), float(torch.quantile(f_, 0.99)),
                   100 * float((sm_ < f_).float().mean())))
        if a.nn_floor_max > 0:
            f_ = dec.cov_floor
            cap = a.nn_floor_max * float(f_.median())
            n_cap = int((f_ > cap).sum())
            with torch.no_grad():
                dec.cov_floor.clamp_(max=cap)
            say('plancher borne a %.1f x la mediane = %.4f : %d gaussiennes ramenees a la borne'
                % (a.nn_floor_max, cap, n_cap))
    edges = None
    if a.w_smooth > 0:
        mov = torch.cat([idx_s, idx_q])
        nbr, dst = [], []
        for i0 in range(0, len(mov), 512):
            d = torch.cdist(X[mov[i0:i0 + 512]], X)
            v_, j_ = d.topk(a.knn + 1, largest=False)
            nbr.append(j_[:, 1:])
            dst.append(v_[:, 1:])
        nbr, dst = torch.cat(nbr), torch.cat(dst)
        src = mov[:, None].expand(-1, a.knn).reshape(-1)
        nbr, dst = nbr.reshape(-1), dst.reshape(-1)
        sig = float(dst.median())
        e_w = torch.exp(-0.5 * (dst / sig) ** 2)
        is_mov = torch.zeros(len(X), dtype=torch.bool, device=dev)
        is_mov[mov] = True
        edges = (src, nbr, e_w)
        say('coherence spatiale : %d aretes (%d plus proches voisines de %d gaussiennes '
            'mobiles), dont %d vers une gaussienne FIXE ; sigma %.4f'
            % (len(src), a.knn, len(mov), int((~is_mov[nbr]).sum()), sig))
    say('champs : s -> %d x 12, (s, q) -> %d x 12 ; etat de la scene fixe z* = %s'
        % (len(idx_s), len(idx_q), z_star.tolist()))

    # ── seance multivue : 3 vues fixes + iphone3 ────────────────────────────────
    rig = mv.load_rig(a.rig)
    views = mv.build_views(rig, cams=list(mv.FIXED), side=a.side, full=True)
    Zall = np.load(HERE / a.z_all)
    T = len(Zall)
    T_tr = T - int(a.val_sec * 60000 / 1001)
    Z = torch.tensor(Zall, device=dev)
    if a.native:
        # images natives des proxys, deja rectifiees, posees dans un carre `a.native` : cote de
        # rendu = cote du carre, K = celle du carre, masque = pixels de l'image rectifiee
        a.side = a.native
        a.undistort = 0
        frames = {v['cam']: JpegBlob(v['cam'], a.native) for v in views}
        for v in views:
            v['K'] = np.array(frames[v['cam']].meta['K'], float)
    else:
        frames = {v['cam']: np.load(HERE / v['cam'] / ('mv_%d.npy' % a.side), mmap_mode='r')
                  for v in views}
    g0 = np.load(HERE / 'iphone0' / 'gripper.npy')
    e0 = mv.misreg_px('iphone0', np.arange(T, dtype=float), np.abs(np.gradient(g0[:, 0])))
    ok_cam = {v['cam']: (e0 < a.i0_max_px) if v['cam'] == 'iphone0' else np.ones(T, bool)
              for v in views}
    E_fix = torch.tensor(np.stack([v['E'] for v in views]), dtype=torch.float32, device=dev)
    K_fix = torch.tensor(np.stack([v['K'] for v in views]), dtype=torch.float32, device=dev)
    poses = json.loads((HERE / a.i3).read_text(encoding='utf-8'))['iphone3_poses']
    p0 = next(iter(poses.values()))
    size3 = [int(x) for x in p0['size']]
    if a.native:
        m3 = JpegBlob('iphone3', a.native)
        K3np = np.array(m3.meta['K'], float)
    else:
        m3 = np.load(HERE / 'iphone3' / ('mv_%d.npy' % a.side), mmap_mode='r')
        K3np = mv.crop_intrinsics(np.array(p0['K'], float), (0, 0, size3[0], size3[1]), a.side)
    K3 = torch.tensor(K3np, dtype=torch.float32, device=dev)
    rows = {'train': [], 'val': []}
    i3_corr = None
    if a.i3_pose_corr:
        f_ = np.load(HERE / a.i3_pose_corr)
        i3_corr = {str(k): M_ for k, M_ in zip(f_['keys'], f_['M'])}
        kb, ka = (('l1_before', 'l1_after') if 'l1_before' in f_.files else ('lp0', 'lp1'))
        say('poses iphone3 : %d corrections par frame lues dans %s (L1 median %.4f -> %.4f)'
            % (len(i3_corr), a.i3_pose_corr, float(np.median(f_[kb])), float(np.median(f_[ka]))))
    i3_bad = set(str(k) for k in np.load(HERE / a.i3_exclude)) if a.i3_exclude else set()
    if i3_bad:
        say('poses iphone3 : %d instants jetes (%s)' % (len(i3_bad), a.i3_exclude))
    for t, pz in poses.items():
        if t in i3_bad:
            continue
        if a.time_fix:
            # CORRIGE (2026-09-30) : l'ancienne conversion lisait l'image iphone3 jusqu'a
            # 7.8 frames (median 3.9) APRES la frame dont la pose a ete calculee, et donnait au
            # decodeur l'etat de cette frame posee : en mouvement, la pince de l'image n'etait
            # pas a c(s). On lit la frame posee et l'etat simultane.
            j3 = (int(pz['frame_i3']) - timebase.PROXY_START['iphone3'] if 'frame_i3' in pz
                  else shown_proxy('iphone3', t))
            i_ref = proxy_to_ref('iphone3', j3)
        else:
            i_ref = timebase.ref_index_of_shown(mv.REF, t)
        if not (0 <= i_ref < T):
            continue
        if not a.time_fix:
            j3 = int(round(mv.ref_to_proxy('iphone3', i_ref)))
        if not (0 <= j3 < len(m3)):
            continue
        Ei = np.eye(4)
        Ei[:3, :3] = np.array(pz['R'], float)
        Ei[:3, 3] = np.array(pz['t'], float).ravel()
        fixed_frame = False
        if i3_corr is not None and t in i3_corr and i_ref < T_tr:
            Ei = i3_corr[t] @ Ei              # correction par frame (refine_i3_poses.py)
            fixed_frame = True                # recalee contre la scene : repere de la scene fixe
        rows['train' if i_ref < T_tr else 'val'].append((int(round(i_ref)), j3, Ei, fixed_frame))
    for k in rows:
        rows[k].sort(key=lambda r: r[0])
    E3 = {k: torch.tensor(np.stack([r[2] for r in rows[k]]), dtype=torch.float32, device=dev)
          for k in rows}
    cams1 = [v['cam'] for v in views] + ['iphone3']
    V = len(cams1)
    # rectification des cibles 256 x 256 (K anisotrope, meme modele de distorsion)
    grids, valids = [], []
    for k, cam in enumerate(cams1):
        Kc = K3np if cam == 'iphone3' else views[k]['K']
        dc = np.array(p0['dist'], float) if cam == 'iphone3' else views[k]['dist']
        if a.native:
            src_ = m3 if cam == 'iphone3' else frames[cam]
            g_, v_ = None, torch.tensor(src_.mask, dtype=torch.float32, device=dev)
        elif a.undistort:
            g_, v_ = undistort_grid(Kc, dc, a.side, a.side, dev)
        else:
            g_, v_ = None, torch.ones(a.side, a.side, device=dev)
        grids.append(g_)
        valids.append(v_)
    valid1 = torch.stack(valids)[None, :, None]                       # (1, V, 1, S, S)
    say('seance multivue : %d instants a pose d iphone3 en entrainement, %d en held-out ; '
        'rectification %s (k1 %s)' % (len(rows['train']), len(rows['val']),
                                      'oui' if a.undistort else 'non',
                                      [round(float(v['dist'][0]), 3) for v in views]
                                      + [round(float(p0['dist'][0]), 3)]))

    def tgt(cam, i_ref):
        j = int(np.clip(round(float(mv.ref_to_proxy(cam, float(i_ref)))), 0,
                        len(frames[cam]) - 1))
        return np.asarray(frames[cam][j], np.float32) / 255.0

    # ── corrections APPRISES des poses (`--learn-poses`) ────────────────────────────────
    # E_corrige = exp(xi) E, xi = (rotation axe-angle, translation), nul a l'init :
    #   - une correction par camera FIXE (3 x 6) ;
    #   - iphone3 : une correction qui varie lentement dans le temps, noeuds tous les
    #     `--i3-knot-sec` s interpoles lineairement (une correction PAR frame ne s'apprendrait
    #     pas : chaque frame n'est tiree que quelques fois) ; instants held-out NON corriges.
    # Les poses de la scene fixe restent figees : elles fixent le repere (sans quoi scene et
    # cameras pourraient deriver ensemble).
    def se3_batch(xi):
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

    fps_ = 60000 / 1001
    pose_fix = nn.Parameter(torch.zeros(len(views), 6, device=dev))
    n_kn3 = int(np.ceil(T / (a.i3_knot_sec * fps_))) + 2
    pose_i3 = nn.Parameter(torch.zeros(n_kn3, 6, device=dev))
    if ck_res is not None and 'pose_fix' in ck_res:
        with torch.no_grad():
            pose_fix.copy_(ck_res['pose_fix'].to(dev))
            # la spline lente est deja incluse dans les corrections par frame : elle repart de 0
            same_corr = (a.i3_pose_corr and ck_res['args'].get('i3_pose_corr') == a.i3_pose_corr)
            if ck_res['pose_i3'].shape == pose_i3.shape and (not a.i3_pose_corr or same_corr):
                pose_i3.copy_(ck_res['pose_i3'].to(dev))

    def sim_world():
        """(Sd, G^-1) : une pose E du repere de la seance devient Sd E G^-1 dans celui de la
        scene fixe (meme rotation de camera composee, translation a l'echelle de la scene fixe)."""
        R, sc = dec.sim_rs()
        t_ = dec.world_sim[4:]
        Gi = torch.eye(4, device=dev)
        Gi = torch.cat([torch.cat([R.T / sc, (-(R.T @ t_) / sc)[:, None]], 1), Gi[3:]], 0)
        Sd = torch.diag(torch.cat([sc.expand(3), torch.ones(1, device=dev)]))
        return Sd, Gi

    if a.global_sim:
        from scipy.spatial.transform import Rotation as Rot_
        with torch.no_grad():
            if bool((dec.world_sim == 0).all()):
                # initialisation par la similitude qui explique au mieux les corrections apprises
                # des cameras fixes (rotation moyenne, puis echelle et translation des centres)
                E0n = E_fix.detach().cpu().double().numpy()
                E1n = (se3_batch(pose_fix) @ E_fix).detach().cpu().double().numpy()
                Rw = Rot_.from_matrix([E1n[k][:3, :3].T @ E0n[k][:3, :3]
                                       for k in range(len(E0n))]).mean().as_matrix()
                Cc0 = np.array([-E[:3, :3].T @ E[:3, 3] for E in E0n])
                C1 = np.array([-E[:3, :3].T @ E[:3, 3] for E in E1n])
                A_ = np.zeros((3 * len(Cc0), 4))
                b_ = C1.ravel()
                for k in range(len(Cc0)):
                    A_[3 * k:3 * k + 3, 0] = Rw @ Cc0[k]
                    A_[3 * k:3 * k + 3, 1:] = np.eye(3)
                x_ = np.linalg.lstsq(A_, b_, rcond=None)[0]
                dec.world_sim.copy_(torch.tensor(np.r_[Rot_.from_matrix(Rw).as_rotvec(),
                                                       np.log(x_[0]), x_[1:]],
                                                 dtype=torch.float32, device=dev))
                Sd, Gi = sim_world()
                Ep = (Sd @ E_fix @ Gi).detach().cpu().double().numpy()
                res_ = np.zeros((len(E0n), 6))
                for k in range(len(E0n)):
                    Rr = E1n[k] @ np.linalg.inv(Ep[k])
                    res_[k, :3] = Rot_.from_matrix(Rr[:3, :3]).as_rotvec()
                    res_[k, 3:] = Rr[:3, 3]
                pose_fix.copy_(torch.tensor(res_, dtype=torch.float32, device=dev))
                say('similitude seance -> scene fixe initialisee sur les corrections des cameras '
                    'fixes : rotation %.2f mrad, echelle %+.2f %%, translation %s ; corrections '
                    'propres restantes %s mrad'
                    % (1e3 * np.linalg.norm(Rot_.from_matrix(Rw).as_rotvec()), 100 * (x_[0] - 1),
                       np.round(x_[1:], 4).tolist(),
                       np.round(1e3 * np.linalg.norm(res_[:, :3], axis=1), 2).tolist()))
                err_ = float((se3_batch(pose_fix) @ (Sd @ E_fix @ Gi) - torch.tensor(
                    E1n, dtype=torch.float32, device=dev)).abs().max())
                say('  controle : ecart max des poses fixes avant / apres reparametrage %.2e' % err_)
            # poses iphone3 deja dans le repere de la scene fixe (recalees par PnP) : ramenees
            # dans celui de la seance par la similitude courante, pour que tout iphone3 la suive
            Sd, Gi = sim_world()
            Sdi, G = torch.linalg.inv(Sd), torch.linalg.inv(Gi)
            for k_ in (E3 if a.global_sim_i3 else []):
                ff = torch.tensor([r[3] for r in rows[k_]], dtype=torch.bool, device=dev)
                if bool(ff.any()):
                    E3[k_][ff] = Sdi @ E3[k_][ff] @ G
            if a.global_sim_i3:
                say('  %d poses iphone3 recalees ramenees dans le repere de la seance ; la '
                    'similitude s applique ensuite a toutes'
                    % sum(int(r[3]) for k_ in rows for r in rows[k_]))
            else:
                say('  similitude appliquee aux cameras fixes et au porteur, pas a l iphone3')

    def corr_i3(frames):
        x = torch.as_tensor(frames, dtype=torch.float32, device=dev) / (a.i3_knot_sec * fps_)
        i0 = x.floor().clamp(0, n_kn3 - 2).long()
        t_ = (x - i0.float())[:, None]
        return se3_batch((1 - t_) * pose_i3[i0] + t_ * pose_i3[i0 + 1])

    def batch1(split, ks):
        rr = [rows[split][k] for k in ks]
        idx = np.array([r[0] for r in rr])
        Ef_, E3_ = E_fix, E3[split][ks]
        if a.global_sim:
            Sd, Gi = sim_world()
            Ef_ = Sd @ E_fix @ Gi
            if a.global_sim_i3:
                E3_ = Sd @ E3_ @ Gi
        if a.learn_poses:
            Ef_ = se3_batch(pose_fix) @ Ef_
            if split == 'train':
                E3_ = corr_i3(idx) @ E3_
        vm = torch.cat([Ef_[None].expand(len(ks), -1, -1, -1), E3_[:, None]], 1)
        Km = torch.cat([K_fix[None].expand(len(ks), -1, -1, -1),
                        K3[None, None].expand(len(ks), 1, -1, -1)], 1)
        tg = torch.tensor(np.stack([np.stack([tgt(v['cam'], i) for v in views]
                                             + [np.asarray(m3[r[1]], np.float32) / 255.0])
                                    for i, r in zip(idx, rr)]), device=dev)   # (B, V, 3, S, S)
        if a.undistort:
            tg = torch.stack([F.grid_sample(tg[:, v], grids[v].expand(len(ks), -1, -1, -1),
                                            mode='bilinear', align_corners=True)
                              for v in range(V)], 1)
        m = torch.tensor(np.stack([[ok_cam[v['cam']][i] for v in views] + [True] for i in idx]),
                         device=dev, dtype=torch.float32)
        return idx, vm, Km, tg, m

    cc_A = nn.Parameter(torch.eye(3, device=dev).repeat(V, 1, 1))
    cc_b = nn.Parameter(torch.zeros(V, 3, device=dev))

    if ck_res is not None:
        with torch.no_grad():
            cc_A.copy_(ck_res['color1']['A'].to(dev))
            cc_b.copy_(ck_res['color1']['b'].to(dev))

    def color1(out):
        return (torch.einsum('vij,bvjhw->bvihw', cc_A, out)
                + cc_b[None, :, :, None, None]).clamp(0, 1)

    Qc = torch.tensor(zones['zone_q']['corners'], dtype=torch.float32, device=dev)   # (8, 3)

    def obj_mask(E4, K3_, W_, H_, shift):
        """Masque (H, W) : enveloppe convexe du pave q translate de `shift`, projete."""
        X = (Qc + shift) @ E4[:3, :3].T + E4[:3, 3]
        m_ = np.zeros((H_, W_), np.uint8)
        if bool((X[:, 2] > 0.05).all()):
            uv = (X[:, :2] / X[:, 2:3]) @ K3_[:2, :2].T + K3_[:2, 2]
            hull = cv2.convexHull(uv.detach().cpu().numpy().astype(np.float32))
            cv2.fillConvexPoly(m_, hull.astype(np.int32), 1)
        return torch.from_numpy(m_).to(dev)

    def weights1(zb, vm, Km):
        """(B, V, 1, S, S) : 1 + (w_obj - 1) x masque de l'objet, par vue."""
        with torch.no_grad():
            sh_ = dec.carrier(zb[:, 0]) - dec.carrier(dec.z_star[None, 0])
            w = torch.ones(zb.shape[0], V, 1, a.side, a.side, device=dev)
            for b_ in range(zb.shape[0]):
                for v_ in range(V):
                    w[b_, v_, 0] += (a.w_obj - 1) * obj_mask(vm[b_, v_], Km[b_, v_], a.side,
                                                             a.side, sh_[b_]).float()
        return w

    def l1_1w(out, tg, m, w):
        d = ((out - tg).abs() * valid1 * w).sum((2, 3, 4)) / ((valid1 * w).sum((2, 3, 4)) * 3)
        return (d * m).sum() / m.sum().clamp(min=1)

    def l1_1(out, tg, m):
        d = ((out - tg).abs() * valid1).sum((2, 3, 4)) / (valid1.sum((2, 3, 4)) * 3)  # (B, V)
        return (d * m).sum() / m.sum().clamp(min=1), (d * m).sum(0) / m.sum(0).clamp(min=1)

    # ── scene fixe : images, poses raffinees, gains (v3) ────────────────────────
    P2 = json.loads((HERE / a.static_poses).read_text(encoding='utf-8'))
    names2, is_val2 = list(sc['names']), np.asarray(sc['is_val'])
    W2, H2 = [int(v) for v in sc['size']]
    K2 = sc['K'].float().to(dev)
    mx, my = cv2.initUndistortRectifyMap(np.array(P2['K'], float),
                                         np.r_[np.array(P2['dist'], float), 0.0], None,
                                         np.array(P2['K'], float), (W2, H2), cv2.CV_32FC1)
    valid2 = torch.tensor(cv2.remap(np.ones((H2, W2), np.float32), mx, my, cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0.999,
                          dtype=torch.float32, device=dev)
    imgs2 = torch.zeros(len(names2), H2, W2, 3, dtype=torch.uint8, device=dev)
    for i, nm in enumerate(names2):
        im = cv2.imread(str(HERE / P2['images_dir'] / nm))[:, :, ::-1]
        imgs2[i] = torch.from_numpy(np.ascontiguousarray(
            cv2.remap(im, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)))
    VM2 = sc['viewmats'].float().to(dev)
    g2A = nn.Parameter(sc['col_A'].float().to(dev).clone())
    g2b = nn.Parameter(sc['col_b'].float().to(dev).clone())
    tr2, va2 = np.flatnonzero(~is_val2), np.flatnonzero(is_val2)
    ep = HERE / a.exclude_poses
    if a.exclude_poses and ep.exists():
        bad = set(json.loads(ep.read_text(encoding='utf-8'))['names'])
        keep2 = np.array([names2[i] not in bad for i in range(len(names2))])
        say('scene fixe : %d images ecartees par %s (%d d entrainement, %d held-out)'
            % (int((~keep2).sum()), a.exclude_poses, int((~keep2[tr2]).sum()),
               int((~keep2[va2]).sum())))
        tr2, va2 = tr2[keep2[tr2]], va2[keep2[va2]]
    if ck_res is not None:
        with torch.no_grad():
            g2A.copy_(ck_res['color2']['A'].to(dev))
            g2b.copy_(ck_res['color2']['b'].to(dev))
    say('scene fixe : %d images %dx%d (%d held-out), rendues a z*' % (len(names2), W2, H2,
                                                                     len(va2)))

    # etat de la scene fixe : q = 0 (languette au repos, certain), s APPRIS si --learn-s-fixed
    # (le s* mesure peut etre faux de quelques pixels d'arc) ; l'ancrage du decodeur reste z*
    s_fix = nn.Parameter(torch.tensor([a.s_star], device=dev))
    if ck_res is not None and 's_fixed' in ck_res:
        with torch.no_grad():
            s_fix.fill_(float(ck_res['s_fixed']))
    zero2 = torch.zeros(2, device=dev)

    def z_fixed():
        return torch.cat([s_fix, zero2])[None]

    def render2(i):
        return dec.render(z_fixed(), VM2[i][None, None], K2[None, None], W2, H2,
                          a.near)[0, 0]                                     # (3, H, W)

    # gaussiennes a surponderer, fixees une fois pour toutes : celles de la zone q
    ind_q = torch.zeros(len(dec.xyz), device=dev)
    ind_q[dec.idx_q] = 1.0

    # ── optimisation ───────────────────────────────────────────────────────────
    opt = torch.optim.Adam([
        {'params': [dec.xyz], 'lr': a.lr_xyz},
        {'params': [dec.log_scale], 'lr': a.lr_scale},
        {'params': [dec.quat], 'lr': a.lr_quat},
        {'params': [dec.logit_alpha], 'lr': a.lr_alpha},
        {'params': [dec.sh0], 'lr': a.lr_sh0},
        {'params': [dec.shN], 'lr': a.lr_shN},
        {'params': list(dec.field_s.parameters()) + list(dec.field_q.parameters()),
         'lr': a.lr_field},
        {'params': [cc_A, cc_b, g2A, g2b], 'lr': a.lr_color},
        {'params': [s_fix], 'lr': a.lr_s_fixed if a.learn_s_fixed else 0.0},
        {'params': [pose_fix, pose_i3], 'lr': a.lr_pose if a.learn_poses else 0.0},
        {'params': [dec.world_sim], 'lr': a.lr_sim if a.global_sim else 0.0},
    ], eps=1e-15)
    FIELD = 6

    # instants held-out au BORD du domaine en q : les 8 plus grands |q| (norme 2) / bord
    # (bord = p99.5 de |q|), espaces d'au moins 60 frames
    qb_ = np.percentile(np.linalg.norm(Zall[:, 1:], axis=1), 99.5)
    iv_ = np.array([r[0] for r in rows['val']])
    rv_ = np.linalg.norm(Zall[iv_, 1:], axis=1) / qb_
    edge_ks = []
    for k_ in np.argsort(-rv_):
        if all(abs(iv_[k_] - iv_[j_]) >= 60 for j_ in edge_ks):
            edge_ks.append(int(k_))
        if len(edge_ks) == 8:
            break
    edge_ks = np.array(edge_ks)
    say('evaluation au bord : 8 instants held-out, |q| / bord de %.2f a %.2f'
        % (rv_[edge_ks].min(), rv_[edge_ks].max()))

    @torch.no_grad()
    def evaluate(step):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        msg = []
        for split in ('train', 'val', 'bord'):
            if split == 'bord':
                ks = edge_ks
                split_rows = 'val'
            else:
                ks = np.linspace(0, len(rows[split]) - 1, 8).astype(int)
                split_rows = split
            tot, per = 0.0, torch.zeros(V, device=dev)
            for i in range(0, 8, 2):
                idx, vm, Km, tg, m = batch1(split_rows, ks[i:i + 2])
                t_, p_ = l1_1(color1(dec.render(Z[idx], vm, Km, a.side, a.side, a.near)), tg, m)
                tot += float(t_) / 4
                per += p_ / 4
            last_l1[split] = tot
            msg.append('%s L1 %.4f (%s)' % (split, tot, ' '.join(
                '%s %.4f' % (c[-1], float(x)) for c, x in zip(cams1, per))))
        ps_raw, ps_fit = [], []
        mk = valid2 > 0
        for i in va2:
            x = render2(i).permute(1, 2, 0)
            y = imgs2[i].float() / 255.0
            ps_raw.append(-10 * math.log10(float(((x - y) ** 2)[mk].mean())))
            xf, yf = x[mk], y[mk]
            xm, ym = xf.mean(0), yf.mean(0)
            gg = ((xf - xm) * (yf - ym)).sum(0) / ((xf - xm) ** 2).sum(0).clamp(min=1e-9)
            ps_fit.append(-10 * math.log10(float((((gg * (xf - xm) + ym).clamp(0, 1) - yf) ** 2)
                                                  .mean())))
        msg.append('s scene fixe %.4f' % float(s_fix.detach()))
        if a.global_sim:
            ws = dec.world_sim.detach()
            msg.append('similitude : rot %.2f mrad, echelle %+.2f %%'
                       % (1e3 * float(ws[:3].norm()), 100 * (float(torch.exp(ws[3])) - 1)))
        if a.learn_poses:
            pf = pose_fix.detach()
            msg.append('poses fixes : rot %s mrad, trans %s' % (
                '/'.join('%.2f' % (1e3 * float(x)) for x in pf[:, :3].norm(dim=1)),
                '/'.join('%.4f' % float(x) for x in pf[:, 3:].norm(dim=1))))
            msg.append('iphone3 : rot mediane %.2f mrad' % (1e3 * float(pose_i3.detach()[:, :3]
                                                                         .norm(dim=1).median())))
        msg.append('scene fixe held-out PSNR %.2f dB brut, %.2f apres gain/biais'
                   % (np.mean(ps_raw), np.mean(ps_fit)))
        # planche : 4 instants (2 train, 2 val) x 4 vues, cible / rendu ; + 2 vues fixes
        ks_t = np.linspace(0, len(rows['train']) - 1, 2).astype(int)
        ks_v = np.linspace(0, len(rows['val']) - 1, 2).astype(int)
        cols = []
        for split, ks in (('train', ks_t), ('val', ks_v)):
            idx, vm, Km, tg, m = batch1(split, ks)
            out = color1(dec.render(Z[idx], vm, Km, a.side, a.side, a.near))
            for b in range(len(ks)):
                cols.append(('%s t=%d s=%.2f q=(%.2f, %.2f)' % (split, idx[b], *Zall[idx[b]]),
                             tg[b], out[b]))
        fig, ax = plt.subplots(2 * V, len(cols), figsize=(3.2 * len(cols), 1.7 * 2 * V),
                               squeeze=False)
        for c, (lab, tg_, out_) in enumerate(cols):
            for r in range(V):
                ax[2 * r][c].imshow(tg_[r].permute(1, 2, 0).cpu().numpy().clip(0, 1))
                ax[2 * r + 1][c].imshow(out_[r].permute(1, 2, 0).cpu().numpy().clip(0, 1))
                for k in (2 * r, 2 * r + 1):
                    ax[k][c].set_xticks([])
                    ax[k][c].set_yticks([])
                if c == 0:
                    ax[2 * r][c].set_ylabel('%s cible' % cams1[r][-1], fontsize=7)
                    ax[2 * r + 1][c].set_ylabel('%s rendu' % cams1[r][-1], fontsize=7)
            ax[0][c].set_title(lab, fontsize=7)
        fig.suptitle('pas %d : %s' % (step, ' | '.join(msg)), fontsize=7)
        fig.tight_layout()
        fig.savefig(run / ('recon_%06d.png' % step), dpi=80)
        plt.close(fig)
        pick = va2[np.linspace(0, len(va2) - 1, 3).astype(int)]
        fig, ax = plt.subplots(2, 3, figsize=(15, 5.5))
        for c, i in enumerate(pick):
            ax[0, c].imshow(render2(i).permute(1, 2, 0).cpu().numpy())
            ax[1, c].imshow(imgs2[i].cpu().numpy())
            ax[0, c].set_title('rendu a z*, %s' % names2[i], fontsize=8)
            ax[1, c].set_title('image held-out', fontsize=8)
            for x_ in ax[:, c]:
                x_.axis('off')
        fig.suptitle('pas %d : scene fixe' % step, fontsize=9)
        fig.tight_layout()
        fig.savefig(run / ('static_%06d.png' % step), dpi=80)
        plt.close(fig)
        return 'eval pas %d : ' % step + ' | '.join(msg)

    last_l1 = {}

    def save(step, path=None):
        torch.save(dict(dec=dec.state_dict(), step=step, args=vars(a), cams1=cams1,
                        color1=dict(A=cc_A.detach().cpu(), b=cc_b.detach().cpu()),
                        color2=dict(A=g2A.detach().cpu(), b=g2b.detach().cpu()),
                        s_fixed=float(s_fix.detach()),
                        pose_fix=pose_fix.detach().cpu(), pose_i3=pose_i3.detach().cpu(),
                        i3_knot_sec=a.i3_knot_sec,
                        zones=zones, n=dict(q=len(idx_q), s=len(idx_s), fixed=n_fix),
                        note='ZonedSplat : gaussiennes de base (etat z*) + field_s(s) sur '
                             'idx_s + field_q(s, q) sur idx_q, ancres a z*'), path or run / 'dec.pt')

    @torch.no_grad()
    def screen_size():
        """Plus grande taille a l'ecran (px) de chaque gaussienne sur les cameras d'entrainement."""
        X_, sm = dec.xyz, torch.exp(dec.log_scale).max(1).values
        sm = torch.sqrt(sm ** 2 + dec.cov_floor ** 2)          # taille effective (plancher)
        cams = [(E_fix[k], float(K_fix[k, 0, 0])) for k in range(len(E_fix))]
        i3 = np.linspace(0, len(E3['train']) - 1, min(a.prune_small_n3, len(E3['train']))).astype(int)
        cams += [(E3['train'][k], float(K3[0, 0])) for k in i3]
        i2s = tr2[np.linspace(0, len(tr2) - 1, min(a.prune_small_n2, len(tr2))).astype(int)]
        cams += [(VM2[k], float(K2[0, 0])) for k in i2s]
        best = torch.zeros(len(X_), device=dev)
        for E_, f_ in cams:
            zc = X_ @ E_[2, :3] + E_[2, 3]
            best = torch.maximum(best, torch.where(zc > a.near, f_ * sm / zc.clamp(min=1e-3),
                                                   torch.zeros_like(zc)))
        return best

    @torch.no_grad()
    def prune(keep):
        """Coupe TOUT aux indices gardes (masque booleen global `keep`)."""
        nonlocal edges, ind_q
        K_old = len(keep)
        newidx = torch.cumsum(keep.long(), 0) - 1                 # ancien -> nouveau (si garde)

        def cut(owner, attr, sel):
            # NOUVEAU parametre (changer .data en place casse autograd apres un premier backward),
            # reporte dans les groupes de l'optimiseur avec ses etats d'Adam coupes au masque sel
            old = getattr(owner, attr)
            new = nn.Parameter(old.data[sel].clone())
            st = opt.state.pop(old, None)
            for g_ in opt.param_groups:
                g_['params'] = [new if q_ is old else q_ for q_ in g_['params']]
            if st:
                opt.state[new] = {k_: (v_[sel].clone() if torch.is_tensor(v_) and v_.dim() > 0
                                       and v_.shape[0] == old.shape[0] else v_)
                                  for k_, v_ in st.items()}
            setattr(owner, attr, new)
        for name_ in ('xyz', 'quat', 'log_scale', 'logit_alpha', 'sh0', 'shN'):
            cut(dec, name_, keep)
        for fld, name in ((dec.field_s, 'idx_s'), (dec.field_q, 'idx_q')):
            idx_ = getattr(dec, name)
            kl = keep[idx_]                                      # gaussiennes locales gardees
            rows = (torch.nonzero(kl, as_tuple=True)[0][:, None] * 12
                    + torch.arange(12, device=dev)).reshape(-1)
            rmask = torch.zeros(fld.head.weight.shape[0], dtype=torch.bool, device=dev)
            rmask[rows] = True
            cut(fld.head, 'weight', rmask)
            cut(fld.head, 'bias', rmask)
            fld.n_gaussians = int(kl.sum())
            if name == 'idx_q':
                dec.q_gate = dec.q_gate[kl].clone()
            setattr(dec, name, newidx[idx_[kl]].clone())
        dec.carrier_w = dec.carrier_w[keep].clone()
        dec.cov_floor = dec.cov_floor[keep].clone()
        if hasattr(dec, '_pos'):
            del dec._pos
        if edges is not None:
            src_, nbr_, w_ = edges
            ke = keep[src_] & keep[nbr_]
            edges = (newidx[src_[ke]], newidx[nbr_[ke]], w_[ke])
        ind_q = torch.zeros(int(keep.sum()), device=dev)
        ind_q[dec.idx_q] = 1.0
        return K_old - int(keep.sum())

    def prune_small(step):
        if a.prune_small_px <= 0 and a.min_px_q <= 0:
            return
        best = screen_size()
        if a.min_px_q > 0:
            # plancher de taille de la languette : agrandir plutot que supprimer (meme forme)
            iq_ = dec.idx_q
            bq = best[iq_]
            small = (bq < a.min_px_q) & (bq > 0)
            with torch.no_grad():
                dec.log_scale.data[iq_[small]] += torch.log(a.min_px_q / bq[small])[:, None]
            best[iq_[small]] = a.min_px_q
            say('  plancher %.1f px de la zone q, pas %d : %d gaussiennes agrandies (facteur median '
                '%.1f)' % (a.min_px_q, step, int(small.sum()),
                           float((a.min_px_q / bq[small]).median()) if small.any() else 1.0))
        keep = best >= a.prune_small_px
        grp = torch.zeros(len(keep), dtype=torch.long, device=dev)
        grp[dec.idx_s] = 1
        grp[dec.idx_q] = 2
        if a.max_gauss > 0 and int(keep.sum()) > a.max_gauss:
            # plafond : on retire les fixes de plus faible contribution (taille ecran x opacite)
            n_cut = int(keep.sum()) - a.max_gauss
            score = best * torch.sigmoid(dec.logit_alpha.detach())
            score[~keep | (grp != 0)] = float('inf')
            keep[torch.argsort(score)[:n_cut]] = False
            say('  plafond %d gaussiennes : %d fixes de plus faible contribution retirees'
                % (a.max_gauss, n_cut))
        nf, ns_, nq_ = (int((~keep & (grp == g)).sum()) for g in range(3))
        n = prune(keep)
        say('  elagage sous %.2f px, pas %d : %d gaussiennes supprimees (fixes %d, zone s %d, '
            'zone q %d), %d restent' % (a.prune_small_px, step, n, nf, ns_, nq_, len(dec.xyz)))

    prune_small(0)
    say(evaluate(0))
    rng = np.random.default_rng(a.seed)
    prob = None
    if a.w_q_edge != 1 or a.w_s_far != 1 or a.fast_v0 > 0:
        it = np.array([r[0] for r in rows['train']])
        zt = Zall[it]
        # norme 2 de q : q0 et q1 ont des unites comparables, q grand dans les deux composantes
        # doit peser plus que q grand dans une seule
        qb = np.percentile(np.linalg.norm(Zall[:, 1:], axis=1), 99.5)
        rq = np.minimum(1.0, np.linalg.norm(zt[:, 1:], axis=1) / qb)
        ds = np.abs(zt[:, 0] - a.s_star)
        w_ = (1 + (a.w_q_edge - 1) * rq) * (1 + (a.w_s_far - 1) * ds / max(ds.max(), 1e-9))
        if a.fast_v0 > 0:
            # mouvement rapide du bras : a 17 px par frame, un decalage de moins d'une frame
            # entre cameras et le rolling shutter suffisent a rendre les quatre vues
            # incoherentes avec un s unique ; poids 1 / (1 + (v / v0)^2), plancher --fast-min.
            # |ds/dt| du suivi de l'iphone1 (px d'arc / s), lisse sur 2 frames
            from scipy.ndimage import gaussian_filter1d
            s_px = np.load(HERE / mv.REF / 's_track.npy').astype(float)
            v_all = np.abs(np.gradient(gaussian_filter1d(s_px, 2))) * 60000 / 1001
            v_ = v_all[np.clip(it, 0, len(v_all) - 1)]
            wv = np.maximum(a.fast_min, 1.0 / (1.0 + (v_ / a.fast_v0) ** 2))
            w_ = w_ * wv
            say('poids de vitesse : v0 %.0f px/s, plancher %.2f ; %.1f %% des instants au plancher, '
                'poids moyen %.2f' % (a.fast_v0, a.fast_min, 100 * float((wv <= a.fast_min).mean()),
                                      float(wv.mean())))
        prob = w_ / w_.sum()
        say('tirage pondere : bord en q |q| = %.2f (p99.5 de la norme 2) ; poids min %.2f, median '
            '%.2f, max %.2f ; 10 %% des instants les plus lourds tires %.1f %% du temps'
            % (qb, w_.min(), np.median(w_), w_.max(),
               100 * float(np.sort(prob)[-len(prob) // 10:].sum())))
    best = {'l1': last_l1.get('val', float('inf'))}
    save(0, run / 'dec_best.pt')
    warm0 = 0
    t0, acc = time.time(), []
    for step in range(1, a.steps + 1):
        ks = (rng.choice(len(rows['train']), size=1, p=prob) if prob is not None
              else rng.integers(0, len(rows['train']), size=1))
        idx, vm, Km, tg, m = batch1('train', ks)
        if a.w_obj != 1:
            out1, cov1 = dec.render(Z[idx], vm, Km, a.side, a.side, a.near, ind=ind_q)
            rec1 = l1_1w(color1(out1), tg, m, 1 + (a.w_obj - 1) * cov1)
        else:
            rec1, _ = l1_1(color1(dec.render(Z[idx], vm, Km, a.side, a.side, a.near)), tg, m)
        i2 = int(rng.choice(tr2))
        y2 = imgs2[i2].float() / 255.0
        v2 = valid2[..., None]
        if a.w_obj != 1:
            o2_, c2_ = dec.render(z_fixed(), VM2[i2][None, None], K2[None, None], W2, H2,
                                  a.near, ind=ind_q)
            out2 = o2_[0, 0].permute(1, 2, 0) * g2A[i2] + g2b[i2]
            v2w = v2 * (1 + (a.w_obj - 1) * c2_[0, 0].permute(1, 2, 0))
        else:
            out2 = render2(i2).permute(1, 2, 0) * g2A[i2] + g2b[i2]
            v2w = v2
        l1_2 = ((out2 - y2).abs() * v2w).sum() / (v2w.sum() * 3)
        ls_2 = 1 - ssim((out2 * v2).permute(2, 0, 1)[None], (y2 * v2).permute(2, 0, 1)[None])
        rec2 = (1 - a.ssim_w) * l1_2 + a.ssim_w * ls_2
        reg2 = ((g2A[i2] - 1) ** 2).sum() + (g2b[i2] ** 2).sum()
        smooth = torch.zeros((), device=dev)
        if edges is not None:
            src, nbr, e_w = edges
            pick = torch.randint(len(src), (a.smooth_edges,), device=dev)
            gi = torch.cat([src[pick], nbr[pick]])
            uniq, inv = torch.unique(gi, return_inverse=True)
            z0 = Z[torch.tensor(idx, device=dev)]
            zz = torch.cat([z0] + [z0 + a.smooth_h * torch.eye(3, device=dev)[k][None]
                                   for k in range(3)])                     # (4, 3)
            mo = dec.motion(zz, uniq)                                      # (4, U, 9)
            J = ((mo[1:] - mo[:1]) / a.smooth_h).permute(1, 0, 2).reshape(len(uniq), 27)
            Ji, Jj = J[inv[:a.smooth_edges]], J[inv[a.smooth_edges:]]
            ww = e_w[pick]
            smooth = (ww * ((Ji - Jj) ** 2).sum(1)).sum() / ww.sum()
        aniso = torch.zeros((), device=dev)
        if a.w_aniso > 0:
            ga = torch.randint(len(dec.xyz), (a.aniso_n,), device=dev)
            ls_ = dec.log_scale[ga]
            lt = math.log(a.aniso_ratio)
            aniso = F.relu(ls_.max(1).values - ls_.min(1).values - lt).pow(2).mean()
            mv_ = ga[dec.carrier_w[ga] >= 0]                       # tous ; filtre ci-dessous
            mv_ = mv_[(dec._pos[0, mv_] >= 0) | (dec._pos[1, mv_] >= 0)] \
                if hasattr(dec, '_pos') else mv_[:0]
            if len(mv_):
                ld = dec.log_scale[mv_] + dec.motion(Z[torch.tensor(idx, device=dev)], mv_)[0, :, 3:6]
                aniso = aniso + F.relu(ld.max(1).values - ld.min(1).values - lt).pow(2).mean()
        big = torch.zeros((), device=dev)
        if a.w_big > 0:
            # taille : demi-axe max au plus --big-ratio x la mediane (detachee) de toutes les
            # gaussiennes ; pour les mobiles, la taille deformee a l'etat courant compte aussi
            lmax_all = dec.log_scale.detach().max(1).values
            lt_big = float(lmax_all.median()) + math.log(a.big_ratio)
            gb = torch.randint(len(dec.xyz), (a.aniso_n,), device=dev)
            big = F.relu(dec.log_scale[gb].max(1).values - lt_big).pow(2).mean()
            if hasattr(dec, '_pos'):
                mb = gb[(dec._pos[0, gb] >= 0) | (dec._pos[1, gb] >= 0)]
                if len(mb):
                    ldb = dec.log_scale[mb] + dec.motion(Z[torch.tensor(idx, device=dev)], mb)[0, :, 3:6]
                    big = big + F.relu(ldb.max(1).values - lt_big).pow(2).mean()
        loss = (rec1 + a.w_static * rec2 + a.color_reg * reg2 + a.w_smooth * smooth
                + a.w_aniso * aniso + a.w_big * big)
        opt.param_groups[FIELD]['lr'] = a.lr_field * min(1.0, (step - warm0) / max(a.warmup, 1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        bad = 0
        for p_ in dec.parameters():
            if p_.grad is not None:
                mm = ~torch.isfinite(p_.grad)
                if mm.any():
                    bad += int(mm.sum())
                    p_.grad[mm] = 0.0
        if bad:
            say('  pas %d : %d composantes de gradient non finies, mises a zero' % (step, bad))
        torch.nn.utils.clip_grad_norm_(list(dec.field_s.parameters())
                                       + list(dec.field_q.parameters()), 5.0)
        opt.step()
        if not a.color_q:
            zero_color_q()        # deja sans gradient ; tenu a zero contre toute derive
        acc.append([rec1.item(), l1_2.item(), 1 - ls_2.item(), smooth.item(), aniso.item(),
                    big.item()])
        if step % a.print_every == 0:
            h = np.mean(acc, 0)
            acc = []
            lm_ = dec.log_scale.detach().max(1).values
            say('pas %6d  multivue L1 %.4f  fixe L1 %.4f SSIM %.3f  lissage %.4f  aniso %.3f  '
                'taille %.4f (%.2f %% > %g x med)  alpha moy %.3f  %.0f s'
                % (step, h[0], h[1], h[2], h[3], h[4], h[5],
                   100 * float((lm_ > lm_.median() + math.log(a.big_ratio)).float().mean()),
                   a.big_ratio, float(torch.sigmoid(dec.logit_alpha).mean()), time.time() - t0))
        if a.prune_small_px > 0 and step % a.prune_small_every == 0:
            prune_small(step)
        if a.snapshot_every and step % a.snapshot_every == 0:
            save(step, run / ('dec_pas%07d.pt' % step))
        if step % a.eval_every == 0 or step == a.steps:
            save(step)
            say(evaluate(step))
            v_ = last_l1.get('val', float('nan'))
            if math.isfinite(v_) and v_ < best['l1']:
                best['l1'] = v_
                save(step, run / 'dec_best.pt')
            elif a.guard > 0 and (not math.isfinite(v_) or v_ > a.guard * best['l1']):
                ckb = torch.load(run / 'dec_best.pt', map_location=dev, weights_only=False)
                dec.load_state_dict(ckb['dec'], strict=False)
                with torch.no_grad():
                    cc_A.copy_(ckb['color1']['A'].to(dev))
                    cc_b.copy_(ckb['color1']['b'].to(dev))
                    g2A.copy_(ckb['color2']['A'].to(dev))
                    g2b.copy_(ckb['color2']['b'].to(dev))
                opt.state.clear()
                warm0 = step
                say('  GARDE-FOU pas %d : L1 held-out %.4f > %.1f x la meilleure (%.4f) : retour a '
                    'dec_best.pt (pas %d), Adam vide' % (step, v_, a.guard, best['l1'], ckb['step']))
    logf.close()


if __name__ == '__main__':
    main()
