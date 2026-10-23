"""
Vues de supervision du decodeur 3D : quelle camera, quelle frame, quelle matrice.

Module, pas un script. Il repond a trois questions, et chacune a un piege.

1. QUELLE FRAME, dans chaque camera, montre le meme instant ?
   L'iphone1 sert d'index de reference, parce que c'est lui qui porte l'encodeur et `s`.
   Les quatre cameras n'ayant pas la meme cadence (cf. `timebase.py`), la frame simultanee
   d'une autre camera est FRACTIONNAIRE. On l'arrondit, et `misreg_px` dit combien coute cet
   arrondi, en pixels de l'image originale, a chaque instant.

2. GARDE-T-ON L'IPHONE0, qui est celui qui derive ? OUI, et c'est mesure. La derive est
   connue exactement, donc il ne reste que l'arrondi : 0.05 px de mediane en pixels
   originaux, 9.3 px au p99, 15.7 px au pire, et 0.02 px sur les 37 % de frames lentes. En
   masquant sa perte au-dela de 2 px on garde 84 % des frames, et aucune cible mal recalee
   n'entre dans l'apprentissage. La bonne facon d'ecarter l'iphone0 n'est donc pas de le
   jeter, c'est de le peser.

3. QUELLE PORTION D'IMAGE. **L'IMAGE ENTIERE**, par defaut. La cible est la scene
   COMPLETE dans le repere de la piece ; le repere suiveur ne sert qu'a l'entree de
   l'encodeur, qui lit la deformation de la languette, pas au decodeur, qui rend la piece.
   Recadrer sur l'objet reviendrait a n'apprendre qu'un decor local, et le nuage de fond,
   qui est la seule partie que la stereo reconstruit bien, serait jete.

   L'image entiere rendue a `side x side` n'est pas carree : la mise a l'echelle est donc
   ANISOTROPE, `fx` et `fy` recoivent des facteurs differents. C'est exact dans le modele
   stenope, et c'est ce que fait `crop_intrinsics`. `object_box` reste disponible pour
   regarder de pres, mais ce n'est pas le mode d'entrainement.

⚠️ RECADRER N'EST PAS ZOOMER. Une fenetre `(x0, y0, w, h)` rendue a `side x side` se traduit
exactement dans les intrinseques : `f <- f * side / w`, `c <- (c - x0) * side / w`. La pose
ne bouge pas. Si on se trompait de convention ici, rien ne leverait d'erreur : le decodeur
apprendrait une scene deformee qui rend juste depuis ces trois vues et faux partout ailleurs,
ce qui ne se verrait qu'au moment de la vue nouvelle.

⚠️ La distorsion radiale des intrinseques n'est PAS appliquee par le rasteriseur. Elle est
faible ici (`k1` de l'ordre de 0.03 a 0.17 sur le grand cote normalise) mais non nulle ; les
images de supervision devraient donc etre RECTIFIEES avant recadrage. `undistort_maps` rend
les cartes pour le faire ; `--no-undistort` permet de mesurer ce que ca change.
"""
import json
from pathlib import Path

import cv2
import numpy as np

import timebase

HERE = Path(__file__).parent.resolve()
FIXED = ('iphone0', 'iphone1', 'iphone2')
REF = 'iphone1'                     # camera qui porte l'encodeur, l'index et `s`
VIDEOS = {'iphone0': 'IMG_5179.MOV', 'iphone1': 'IMG_0007.MOV',
          'iphone2': 'IMG_0004.MOV', 'iphone3': 'IMG_0003.MOV'}


# ────────────────────────────── temps ──────────────────────────────

def ref_to_proxy(cam, i_ref):
    """Indice (FRACTIONNAIRE) de frame du proxy de `cam` simultane a la frame `i_ref` du
    proxy de la camera de reference. Passe par l'iphone0, qui est la reference de
    `timebase` : j_cam = i0 (1 + A_cam) + B_cam, donc i0 = (j_ref - B_ref) / (1 + A_ref)."""
    i_ref = np.asarray(i_ref, float)
    i0 = (i_ref - timebase.B[REF]) / (1.0 + timebase.A[REF])
    return i0 * (1.0 + timebase.A[cam]) + timebase.B[cam]


def source_frame(cam, i_ref):
    """Frame de la VIDEO SOURCE de `cam`, arrondie, et la partie fractionnaire perdue."""
    j = ref_to_proxy(cam, i_ref) + timebase.PROXY_START[cam]
    r = np.round(j)
    return r.astype(np.int64), j - r


def misreg_px(cam, i_ref, speed_px_per_frame):
    """Erreur de recalage due a l'arrondi, en pixels de l'image ORIGINALE.

    `speed_px_per_frame` : vitesse de l'objet dans le proxy de `cam`, une valeur par frame de
    ce proxy. Le facteur 2 ramene du proxy 540 a l'original 1080."""
    j, frac = source_frame(cam, i_ref)
    jj = np.clip(ref_to_proxy(cam, i_ref), 0, len(speed_px_per_frame) - 1)
    v = np.interp(jj, np.arange(len(speed_px_per_frame)), speed_px_per_frame)
    return 2.0 * np.abs(frac) * v


# ────────────────────────────── geometrie ──────────────────────────────

def load_rig(path='rig_manual.json'):
    return json.loads((HERE / path).read_text(encoding='utf-8'))


def object_box(rig, cam, grip_id=10, pad=(256, 76, 436), ref_f=None):
    """Fenetre fixe couvrant l'objet sur TOUTE la course, en pixels originaux de `cam`.

    C'est l'UNION des fenetres suiveuses : on reprojette le point de pince (`grip_id`, le
    mobile le plus pointe et celui qui parcourt toute la course) a tous les instants
    triangules, on prend sa boite englobante, et on l'elargit de la demi-fenetre suiveuse.

    `pad = (cote, haut, bas)` en pixels de l'IPHONE1, reprise telle quelle de
    `make_object_crops.py` : fenetre 512 centree sur la pince decalee de 180 vers le bas,
    soit 256 de chaque cote, 256 - 180 = 76 au-dessus et 256 + 180 = 436 en dessous. La
    marge est mise a l'echelle de la focale de chaque camera, les trois etant a des
    distances voisines de l'objet mais pas a la meme focale (1832 contre 1632 et 1643).

    ⚠️ Prendre TOUS les points mobiles au lieu du seul point de pince ouvre la fenetre sur
    tout le bras, socle compris, et la resolution utile s'effondre : mesure, 1013 x 737 px
    au lieu de la fenetre utile.
    """
    c = rig['cameras'][cam]
    K = np.array(c['K'], float)
    R = np.array(c['R'], float)
    t = np.array(c['t'], float).ravel()
    W, H = int(c['size'][0]), int(c['size'][1])
    suffix = '_%d' % grip_id
    X = np.array([v for k, v in rig['points3d'].items()
                  if k.startswith('M_') and k.endswith(suffix)], float)
    if len(X) < 5:
        raise SystemExit('point de pince %d absent de rig_manual.json' % grip_id)
    Xc = R @ X.T + t[:, None]
    p = K @ Xc[:, Xc[2] > 1e-6]
    p = (p[:2] / p[2]).T
    lo = np.percentile(p, 1, axis=0)
    hi = np.percentile(p, 99, axis=0)
    f = K[0, 0] / float(ref_f if ref_f else K[0, 0])
    ps, pu, pd = (np.array(pad, float) * f).astype(int)
    x0 = max(0, int(lo[0]) - ps)
    x1 = min(W, int(hi[0]) + ps)
    y0 = max(0, int(lo[1]) - pu)
    y1 = min(H, int(hi[1]) + pd)
    return x0, y0, x1 - x0, y1 - y0


def crop_intrinsics(K, box, side):
    """Intrinseques d'une fenetre `(x0, y0, w, h)` rendue a `side x side`.

    Le recadrage translate le point principal, la mise a l'echelle multiplie focale ET point
    principal. Les deux axes ont leur propre facteur quand la fenetre n'est pas carree.
    """
    x0, y0, w, h = box
    sx, sy = side / float(w), side / float(h)
    Kc = K.copy()
    Kc[0, 0] *= sx
    Kc[1, 1] *= sy
    Kc[0, 2] = (K[0, 2] - x0) * sx
    Kc[1, 2] = (K[1, 2] - y0) * sy
    return Kc


def extrinsics(rig, cam):
    """Matrice 4x4 monde -> camera."""
    c = rig['cameras'][cam]
    E = np.eye(4)
    E[:3, :3] = np.array(c['R'], float)
    E[:3, 3] = np.array(c['t'], float).ravel()
    return E


def build_views(rig, cams=FIXED, side=256, full=True, grip_id=10, pad=(256, 76, 436)):
    """(liste de dicts) une entree par camera : box, E (4,4), K (3,3) a `side x side`.

    `full=True` (defaut) prend l'IMAGE ENTIERE : la cible est la scene complete dans le
    repere de la piece. `full=False` recadre sur l'objet, ce qui ne sert qu'a regarder.
    """
    ref_f = float(np.array(rig['cameras'][REF]['K'], float)[0, 0])
    out = []
    for cam in cams:
        W, H = (int(v) for v in rig['cameras'][cam]['size'])
        box = (0, 0, W, H) if full else object_box(rig, cam, grip_id, pad, ref_f)
        K = np.array(rig['cameras'][cam]['K'], float)
        out.append(dict(cam=cam, box=box, side=side,
                        E=extrinsics(rig, cam),
                        K=crop_intrinsics(K, box, side),
                        dist=np.array(rig['cameras'][cam]['dist'], float),
                        size=tuple(int(v) for v in rig['cameras'][cam]['size'])))
    return out


def undistort_maps(view):
    """Cartes de rectification a appliquer A L'IMAGE SOURCE avant recadrage."""
    W, H = view['size']
    K = np.array(load_rig()['cameras'][view['cam']]['K'], float)
    d = view['dist']
    d4 = np.zeros(5)
    d4[:len(d)] = d
    return cv2.initUndistortRectifyMap(K, d4, None, K, (W, H), cv2.CV_16SC2)


def crop_frame(img, view, undistort_map=None):
    """Applique la rectification puis la fenetre, et ramene a `side x side`."""
    if undistort_map is not None:
        img = cv2.remap(img, undistort_map[0], undistort_map[1], cv2.INTER_LINEAR)
    x0, y0, w, h = view['box']
    sub = img[y0:y0 + h, x0:x0 + w]
    return cv2.resize(sub, (view['side'], view['side']), interpolation=cv2.INTER_AREA)
