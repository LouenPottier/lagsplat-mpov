"""
Base de temps CORRIGEE des quatre cameras. A importer partout ou une frame est lue.

CE QU'ELLE REMPLACE. Les scripts de ce dossier calculaient la frame d'une camera par
`frame = (t + OFFSET[cam]) * 60000/1001`, c'est-a-dire une cadence commune et un decalage
constant. Les deux parties sont fausses :

  - la CADENCE n'est pas commune. Mesuree sur les horodatages du conteneur, l'iphone0
    tourne a 59.955142 fps et les trois autres a 59.9399, soit un ecart de 254 ppm ;
  - le DECALAGE n'est donc pas constant : il derive de 9 frames sur les 605 s de la
    tranche commune. `iphone0/proxy96.mp4` compte d'ailleurs 36 272 frames la ou les trois
    autres en comptent 36 263, pour la meme tranche demandee.

Ce que cela coutait : a 7 px/frame, la vitesse d'un a-coup du bras, 9 frames valent 60 px
dans l'image. C'est ce qui faisait conclure a tort que la position de la pince dans
l'iphone0 etait aveugle a la profondeur (cf. `curvilinear_s.py` et `measure_drift.py`).

CONVENTION, inchangee : le temps commun `t` reste l'etiquette historique
`t = 60 + i / 59.940060`, ou `i` est l'indice de frame du proxy de l'iphone0. Ce n'est PAS
une seconde reelle, c'est un nom d'instant, et tous les fichiers de pointage deja ecrits
l'emploient. Ce module ne le change pas ; il corrige seulement la frame que chaque camera
doit montrer pour cet instant.

Les quatre videos sont strictement a cadence constante, sans aucune frame sautee (verifie :
zero intervalle superieur a 1.5 frame sur les quatre), donc un modele affine par camera est
exact et non une approximation.

Usage :
    import timebase
    f = timebase.source_frame('iphone3', t)     # frame de la video source, float
    i = timebase.ref_index(t)                   # indice de frame du proxy de l iphone0
"""
import json
from pathlib import Path

HERE = Path(__file__).parent.resolve()
FPS_LABEL = 60000 / 1001          # cadence qui DEFINIT l'etiquette de temps commun
REF = 'iphone0'

# Frame source correspondant a la frame 0 du proxy, telle que `make_proxy.py` l'a coupee.
# (t0_common = 60 s, `-ss` apres `-i` donc coupe exacte a la frame, `-fps_mode passthrough`
# donc aucune frame regularisee.)
#
# ⚠️ CE N'EST PAS `round((60 + OFFSET) * FPS_LABEL)`, la valeur que `make_proxy.py` IMPRIME.
# ffmpeg coupe a la premiere frame dont l'horodatage atteint `-ss`, donc a la cadence REELLE
# de la camera, alors que ce nombre imprime emploie la cadence nominale. L'ecart vaut 2
# frames sur l'iphone0 (3598 contre 3596) et 1 sur les trois autres. Les valeurs ci-dessous
# sont lues sur `<cam>/pts.npy` : premier indice tel que `pts >= 60 + OFFSET[cam]`.
PROXY_START = {'iphone0': 3598, 'iphone1': 3761, 'iphone2': 3221, 'iphone3': 3055}

_drift = json.loads((HERE / 'drift.json').read_text(encoding='utf-8'))

#: cadence mesuree sur les horodatages, par camera
FPS = {c: _drift['cams'][c]['fps'] for c in _drift['cams']}
#: modele affine du decalage, en frames de la camera : j = i * (1 + A) + B
A = {c: _drift['cams'][c]['a'] for c in _drift['cams']}
B = {c: _drift['cams'][c]['b'] for c in _drift['cams']}


def ref_index(t):
    """Indice de frame du proxy de l'iphone0 pour l'etiquette de temps commun `t`."""
    return (float(t) - 60.0) * FPS_LABEL


def proxy_index(cam, t):
    """Indice de frame du proxy de `cam`. Fractionnaire : c'est voulu, les horloges ne
    tombent pas sur les memes instants et arrondir coute jusqu'a une demi-frame."""
    i = ref_index(t)
    return i * (1.0 + A[cam]) + B[cam]


def source_frame(cam, t):
    """Frame de la VIDEO SOURCE de `cam` pour l'etiquette `t`."""
    return PROXY_START[cam] + proxy_index(cam, t)


def legacy_source_frame(cam, t, offset):
    """Ce que calculait l'ancien code, pour mesurer l'ecart. `offset` = make_proxy.OFFSET."""
    return (float(t) + offset) * FPS_LABEL


#: decalage source de `make_proxy.py`, celui-la meme qu'emploie `click_points.py`
OFFSET = {'iphone0': 0.0, 'iphone1': 2.736, 'iphone2': -6.273, 'iphone3': -9.042}


def ref_index_of_shown(cam, t):
    """Indice de frame du proxy de l'iphone0 VRAIMENT simultane a la frame que
    `click_points.py` a AFFICHEE pour `cam` a l'etiquette `t`.

    ⚠️ Ce n'est pas `ref_index(t)`, et l'ecart n'est pas negligeable. `click_points.py`
    affiche la frame source `round((t + OFFSET[cam]) * FPS_LABEL)`, alors que la frame 0 du
    proxy est `PROXY_START[cam]`, lui-meme arrondi. Il reste donc, par camera, une constante
    sous-frame qui vaut jusqu'a une demi-frame, c'est-a-dire 3.5 px a la vitesse d'un a-coup.
    On inverse ici la relation exacte
        frame source de `cam` = PROXY_START[cam] + i * (1 + A[cam]) + B[cam].
    """
    fsrc = round((float(t) + OFFSET[cam]) * FPS_LABEL)
    return (fsrc - PROXY_START[cam] - B[cam]) / (1.0 + A[cam])
