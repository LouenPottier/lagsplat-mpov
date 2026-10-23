"""
DeformSplatDecoder : gaussiennes 2D canoniques + champ de deformation MLP en (q, u).

Difference avec `models_2pt.GaussianSplatDecoder2pt`, qui est le decodeur de reference du
pipeline. Celui-la place des gaussiennes dans l espace joint (x, y, z) et rend la COUPE a
z = q : le centre conditionnel est alors AFFINE en q et la covariance conditionnelle n en
depend pas du tout. C est ce qui donne la forme close de J = d mu / dq, et c est aussi ce
qui borne sa capacite : une gaussienne ne peut que se translater lineairement, elle ne peut
ni tourner ni s allonger quand l objet se plie.

Ici chaque gaussienne a des parametres CANONIQUES, et un reseau commun produit leur
deformation en fonction de l etat :

    mu(q,u)     = mu0     + dmu(q,u)          centre, en coordonnees normalisees
    prof(q,u)   = prof0   + dprof(q,u)        profondeur, qui donne l ORDRE de compositing
    log s(q,u)  = log s0  + dlogs(q,u)        demi-axes
    theta(q,u)  = theta0  + dtheta(q,u)       orientation
    c(q,u)      = c0      + dc(q,u)           couleur
    alpha       = alpha0                      opacite, NON deformee (voir plus bas)

Trois choix structurels, chacun pour une raison precise.

OCCLUSION. Les gaussiennes portent une profondeur, deformable elle aussi, et le rendu les
place a Z = Z0 + prof devant une camera fictive. gsplat trie par profondeur et composite en
alpha front-to-back, donc une gaussienne devant une autre la CACHE, au lieu de se melanger a
elle. C est ce qui manque au decodeur 2pt, ou toutes les gaussiennes sont au meme Z et ou
l ordre de compositing ne veut rien dire. Le placement compense exactement la perspective
(X = (mu_x - 1/2) Z, echelles monde multipliees par Z) de sorte que la position et la taille
PROJETEES ne dependent pas de la profondeur : celle-ci ne sert qu a ordonner.

AUCUNE GAUSSIENNE NE PEUT DISPARAITRE. L opacite ne depend pas de (q, u) et est bornee par
le bas (`alpha_min`), les demi-axes sont bornes par le bas (`scale_min`), et il n y a ni
elagage ni densification. Une gaussienne qui s efface est une gaussienne dont le gradient
meurt : elle ne revient jamais, et la capacite perdue ne se recupere pas. La consequence a
accepter est qu une zone de l image doit etre peinte par la COULEUR des gaussiennes qui s y
trouvent, pas par leur absence.

LE RESEAU EST COMMUN A TOUTES LES GAUSSIENNES. Son entree est l encodage de Fourier de la
position canonique et de l etat, donc le champ de deformation est lisse en espace par
construction : deux gaussiennes voisines recoivent des deformations voisines. C est la
recette de Deformable 3D Gaussians (Yang et al., CVPR 2024), et c est ce qui evite les
20 000 champs independants qu une table de parametres par gaussienne produirait.

Regularisation : voir `q_sensitivity`. Elle mesure, en PIXELS, de combien le contour a
1 sigma de chaque gaussienne se deplace quand q bouge d une unite, et c est cette quantite
que l entrainement penalise, plus fort hors de la zone de l objet.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _fourier(x: torch.Tensor, n_freq: int) -> torch.Tensor:
    """Encodage de Fourier [x, sin(2^k pi x), cos(2^k pi x)] ; n_freq = 0 -> identite."""
    if n_freq <= 0:
        return x
    f = 2.0 ** torch.arange(n_freq, device=x.device, dtype=x.dtype) * math.pi
    a = x.unsqueeze(-1) * f                                  # (..., D, n_freq)
    a = a.flatten(-2)                                        # (..., D * n_freq)
    return torch.cat([x, torch.sin(a), torch.cos(a)], dim=-1)


def _fourier_dim(d: int, n_freq: int) -> int:
    return d if n_freq <= 0 else d * (1 + 2 * n_freq)


class DeformField(nn.Module):
    """
    MLP commun (position canonique, etat) -> deformation.

    La premiere couche est SEPAREE en une part position et une part etat. La part position
    ne depend pas du lot, donc elle se calcule une fois par pas au lieu de B fois, et le
    tenseur d entree (B, K, F) n est jamais materialise. A B = 8 et K = 20 000 c est le
    tenseur le plus gros du graphe, donc ce n est pas une micro-optimisation.
    """

    def __init__(self, cond_dim: int, n_out: int, hidden: int = 128, layers: int = 4,
                 pe_pos: int = 6, pe_cond: int = 4):
        super().__init__()
        self.pe_pos, self.pe_cond = pe_pos, pe_cond
        f_pos = _fourier_dim(2, pe_pos)
        f_cond = _fourier_dim(cond_dim, pe_cond)
        self.lin_pos = nn.Linear(f_pos, hidden)
        self.lin_cond = nn.Linear(f_cond, hidden, bias=False)
        self.body = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers - 1))
        self.head = nn.Linear(hidden, n_out)
        # Sortie nulle a l initialisation : le decodeur part EXACTEMENT de sa configuration
        # canonique, donc la premiere epoque ajuste une scene statique avant d y mettre du
        # mouvement. Un champ non nul au depart brouillerait les deux.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, pos: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """pos (K, 2), cond (B, C) -> (B, K, n_out)"""
        h_pos = self.lin_pos(_fourier(pos, self.pe_pos))          # (K, H)
        h_cond = self.lin_cond(_fourier(cond, self.pe_cond))      # (B, H)
        h = F.silu(h_pos.unsqueeze(0) + h_cond.unsqueeze(1))      # (B, K, H)
        for lin in self.body:
            h = F.silu(lin(h))
        return self.head(h)


class DeformSplatDecoder(nn.Module):
    """
    q, u -> image. Voir la docstring du module.

    `cond_dim` est la dimension de l etat COMPLET passe au champ, `latent_dim` celle de sa
    partie `q`, placee en tete. Seule cette partie est regularisee : `u` est une entree
    mesuree, on ne demande pas au decodeur d en dependre le moins possible, au contraire.
    """

    # bornes des sorties du champ, en unites de chaque parametre
    MAX_DMU = 0.25        # coordonnees normalisees (0.25 = 120 px a 480)
    MAX_DPROF = 0.20
    MAX_DLOGS = 1.0       # facteur e sur les demi-axes
    MAX_DTHETA = math.pi
    MAX_DCOLOR = 0.5

    def __init__(self, latent_dim: int, cond_dim: int, n_gaussians: int,
                 img_size: tuple, n_channels: int = 3,
                 hidden: int = 128, layers: int = 4, pe_pos: int = 6, pe_cond: int = 4,
                 alpha_min: float = 0.02, scale_min: float = 3.e-4,
                 depth_scale: float = 1.0, deform_color: bool = True,
                 deform_depth: bool = True):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.n_gaussians = n_gaussians
        self.img_size = img_size
        self.n_channels = n_channels
        self.alpha_min = alpha_min
        self.scale_min = scale_min
        self.depth_scale = depth_scale
        self.deform_color = deform_color
        # deform_depth = False : la profondeur reste canonique, donc l ordre de compositing
        # ne depend pas de l etat (cas d une scene sans occlusion variable, ou un objet est
        # toujours devant le fond). Le canal reste dans la sortie du champ, ignore.
        self.deform_depth = deform_depth

        K, C = n_gaussians, n_channels
        self.mu_raw = nn.Parameter(torch.zeros(K, 2))
        self.prof_raw = nn.Parameter(torch.zeros(K))
        self.logs_raw = nn.Parameter(torch.zeros(K, 2))
        self.theta_raw = nn.Parameter(torch.zeros(K))
        self.log_alpha = nn.Parameter(torch.zeros(K))
        self.color_raw = nn.Parameter(torch.zeros(K, C))

        self.n_out = 2 + 1 + 2 + 1 + (C if deform_color else 0)
        self.field = DeformField(cond_dim, self.n_out, hidden=hidden, layers=layers,
                                 pe_pos=pe_pos, pe_cond=pe_cond)

    # ── parametres canoniques ─────────────────────────────────────────────

    def canonical(self):
        mu0 = torch.sigmoid(self.mu_raw) * 1.2 - 0.1                 # (K, 2)
        prof0 = torch.tanh(self.prof_raw) * self.depth_scale         # (K,)
        logs0 = self.logs_raw                                        # (K, 2)
        theta0 = self.theta_raw                                      # (K,)
        alpha = self.alpha_min + (1.0 - self.alpha_min) * torch.sigmoid(self.log_alpha)
        color0 = self.color_raw                                      # (K, C) logits
        return mu0, prof0, logs0, theta0, alpha, color0

    def _split(self, out: torch.Tensor):
        """Sorties brutes du champ -> deformations bornees."""
        C = self.n_channels
        dmu = torch.tanh(out[..., 0:2]) * self.MAX_DMU
        dprof = (torch.tanh(out[..., 2]) * self.MAX_DPROF if self.deform_depth
                 else out.new_zeros(out.shape[:-1]))
        dlogs = torch.tanh(out[..., 3:5]) * self.MAX_DLOGS
        dtheta = torch.tanh(out[..., 5]) * self.MAX_DTHETA
        if self.deform_color:
            dcolor = torch.tanh(out[..., 6:6 + C]) * self.MAX_DCOLOR
        else:
            dcolor = out.new_zeros(out.shape[:-1] + (C,))
        return dmu, dprof, dlogs, dtheta, dcolor

    def _field_out(self, pos: torch.Tensor, cond: torch.Tensor, idx=None):
        """Sortie brute du champ pour les gaussiennes de positions `pos` (indices `idx`)."""
        return self.field(pos, cond)

    def deform(self, cond: torch.Tensor, mu0: torch.Tensor = None):
        """cond (B, cond_dim) -> deformations bornees, chacune (B, K, ...)."""
        if mu0 is None:
            mu0 = torch.sigmoid(self.mu_raw) * 1.2 - 0.1
        return self._split(self.field(mu0.detach(), cond))

    def _deform_alpha(self, cond: torch.Tensor, mu0: torch.Tensor):
        """Deformations bornees et logit d opacite ajoute (B, K), None ici : opacite constante."""
        return self.deform(cond, mu0), None

    # ── etat deforme ──────────────────────────────────────────────────────

    def state(self, cond: torch.Tensor):
        """
        Retourne (mu, prof, s, theta, alpha, color) pour un lot d etats.
            mu    (B, K, 2)  coordonnees normalisees
            prof  (B, K)     profondeur (ordre de compositing)
            s     (B, K, 2)  demi-axes, coordonnees normalisees, plancher `scale_min`
            theta (B, K)     orientation
            alpha (K,)       opacite, constante en (q, u) et bornee par le bas
                  (B, K)     si le decodeur deforme l opacite (SharedDeformSplatDecoder)
            color (B, K, C)  dans [0, 1]
        """
        mu0, prof0, logs0, theta0, alpha, color0 = self.canonical()
        (dmu, dprof, dlogs, dtheta, dcolor), dla = self._deform_alpha(cond, mu0)
        if dla is not None:
            alpha = self.alpha_min + (1.0 - self.alpha_min) * torch.sigmoid(
                self.log_alpha.unsqueeze(0) + dla)
        mu = mu0.unsqueeze(0) + dmu
        prof = prof0.unsqueeze(0) + dprof
        s = self.scale_min + F.softplus(logs0.unsqueeze(0) + dlogs)
        theta = theta0.unsqueeze(0) + dtheta
        color = torch.sigmoid(color0.unsqueeze(0) + dcolor)
        return mu, prof, s, theta, alpha, color

    # ── rendu ─────────────────────────────────────────────────────────────

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        """cond (B, cond_dim) -> image (B, C, H, W) dans [0, 1]."""
        from gsplat import rasterization

        H, W = self.img_size
        C = self.n_channels
        K = self.n_gaussians
        mu, prof, s, theta, alpha, color = self.state(cond)
        B = mu.shape[0]
        dev, dt = mu.device, mu.dtype

        Z = 1.0 + prof                                             # (B, K)
        # quaternion wxyz d une rotation d angle theta autour de l axe optique
        half = 0.5 * theta
        quats = torch.stack([torch.cos(half), torch.zeros_like(half),
                             torch.zeros_like(half), torch.sin(half)], dim=-1)  # (B,K,4)

        K_cam = torch.tensor([[float(W), 0.0, W / 2.0],
                              [0.0, float(H), H / 2.0],
                              [0.0, 0.0, 1.0]], device=dev, dtype=dt)
        viewmat = torch.eye(4, device=dev, dtype=dt)
        bg = torch.zeros(C, device=dev, dtype=dt)
        eps_z = 1.e-3

        out = []
        for b in range(B):
            zb = Z[b]                                              # (K,)
            # Position 3D : X = (mu_x - 1/2) Z, donc u = W X / Z + W/2 = W mu_x, sans
            # dependance a la profondeur. Meme chose en Y.
            xyz = torch.stack([(mu[b, :, 0] - 0.5) * zb,
                               (mu[b, :, 1] - 0.5) * zb, zb], dim=1)   # (K, 3)
            # Echelles MONDE = echelles image x Z : la projection divise par Z, donc
            # l empreinte projetee est exactement `s`, quelle que soit la profondeur.
            scales = torch.cat([s[b] * zb.unsqueeze(1),
                                torch.full((K, 1), eps_z, device=dev, dtype=dt)], dim=1)
            renders, _, _ = rasterization(
                means=xyz, quats=quats[b], scales=scales,
                opacities=alpha[b] if alpha.dim() == 2 else alpha, colors=color[b],
                viewmats=viewmat.unsqueeze(0), Ks=K_cam.unsqueeze(0),
                width=W, height=H, near_plane=0.1, far_plane=5.0,
                sh_degree=None, render_mode='RGB', backgrounds=bg,
                eps2d=0.3,
            )
            out.append(renders[0, :, :, :C].permute(2, 0, 1).clamp(0, 1))
        return torch.stack(out, dim=0)

    # ── sensibilite a q, en pixels ────────────────────────────────────────

    def q_sensitivity(self, cond: torch.Tensor, idx: torch.Tensor = None,
                      h: float = 0.05):
        """
        De combien le contour a 1 sigma de chaque gaussienne se deplace, EN PIXELS, quand q
        varie d une unite. C est la quantite que l entrainement penalise.

        Trois canaux, tous ramenes au meme pixel pour pouvoir etre additionnes :
            centre      |d mu / dq| x cote
            demi-axes   |d log s / dq| x s x cote     (le contour se dilate d autant)
            orientation |d theta / dq| x s_max x cote (il tourne d autant, en bout d axe)
        La couleur est rendue a part, n etant pas un deplacement : elle a son propre poids
        dans la loss.

        Derivee par differences FINIES CENTREES sur chaque coordonnee de q, et non par
        autograd : `d` evaluations supplementaires du champ suffisent, la memoire reste
        celle d un forward, et le pas `h` fixe explicitement l ECHELLE a laquelle la
        variation est mesuree. Une derivee exacte en un point ne dit rien de ce qui se passe
        a une demi unite de la, et c est bien la variation a l echelle de q qui nous
        interesse.

        cond (B, cond_dim), idx (M,) sous-ensemble de gaussiennes (None = toutes).
        Retourne (dep, dcol), chacun (B, M), en pixels et en unites de couleur.
        """
        d = self.latent_dim
        H, W = self.img_size
        side = 0.5 * (H + W)
        mu0, _, logs0, _, _, _ = self.canonical()
        if idx is not None:
            mu0 = mu0[idx]
            logs0 = logs0[idx]
        pos = mu0.detach()
        s0 = self.scale_min + F.softplus(logs0)                   # (M, 2)

        dep2 = torch.zeros(cond.shape[0], pos.shape[0],
                           device=cond.device, dtype=cond.dtype)
        dcol2 = torch.zeros_like(dep2)
        for a in range(d):
            e = torch.zeros_like(cond)
            e[:, a] = h
            gp = self._split(self._field_out(pos, cond + e, idx))
            gm = self._split(self._field_out(pos, cond - e, idx))
            inv = 1.0 / (2.0 * h)
            g_mu = (gp[0] - gm[0]) * inv                          # (B, M, 2)
            g_logs = (gp[2] - gm[2]) * inv                        # (B, M, 2)
            g_th = (gp[3] - gm[3]) * inv                          # (B, M)
            g_c = (gp[4] - gm[4]) * inv                           # (B, M, C)

            px_mu = g_mu.pow(2).sum(-1)
            px_s = (g_logs * s0.unsqueeze(0)).pow(2).sum(-1)
            px_th = (g_th * s0.max(dim=1).values.unsqueeze(0)).pow(2)
            dep2 = dep2 + (px_mu + px_s + px_th) * (side ** 2)
            dcol2 = dcol2 + g_c.pow(2).sum(-1)
        # `sqrt(x + eps)` et non `sqrt(x)` : a l initialisation le champ sort exactement
        # zero, donc `dep2 = 0`, ou la derivee de la racine est infinie et le gradient du
        # champ sort NaN des le premier pas. L epsilon vaut (1e-3 px)² d un cote et
        # (1e-4 unite de couleur)² de l autre, tres en dessous de ce qu on mesure.
        return (dep2 + 1.e-6).sqrt(), (dcol2 + 1.e-8).sqrt()

    # ── initialisation ────────────────────────────────────────────────────

    @torch.no_grad()
    def smart_init(self, ref_image: torch.Tensor, sigma_xy: float = 0.006,
                   grad_weight: float = 0.8, zone: np.ndarray = None,
                   zone_frac: float = 0.5):
        """
        Positions tirees sur la carte de gradient de `ref_image`, couleur du pixel tire,
        demi-axes a `sigma_xy`, orientations et profondeurs aleatoires.

        `zone` (H, W) booleen, `zone_frac` : part des gaussiennes reservee a la zone de
        l objet. Sans cela, l objet recevrait sa part en SURFACE (29 % des pixels), alors
        que c est la seule partie de la scene qui se deforme, donc la seule qui a besoin de
        primitives fines. La reservation ne les y enferme pas, elles restent libres de
        migrer.
        """
        K, C = self.n_gaussians, self.n_channels
        img = ref_image.permute(1, 2, 0) if ref_image.shape[0] in (1, 3) else ref_image
        Hi, Wi, _ = img.shape
        gray = img.mean(-1)
        gx = F.pad(gray[:, 1:] - gray[:, :-1], (0, 1))
        gy = F.pad(gray[1:, :] - gray[:-1, :], (0, 0, 0, 1))
        base = grad_weight * (gx ** 2 + gy ** 2).sqrt().flatten() + (1 - grad_weight)
        base = base.cpu().numpy().astype(np.float64)

        def draw(p, n):
            p = p / p.sum()
            return np.random.choice(Hi * Wi, size=n, replace=n > (p > 0).sum(), p=p)

        if zone is not None and 0 < zone_frac < 1:
            z = torch.from_numpy(zone.astype(np.float32))
            z = F.interpolate(z[None, None], size=(Hi, Wi), mode='nearest')[0, 0]
            z = z.numpy().astype(bool).flatten()
            n_in = int(round(zone_frac * K))
            flat = np.concatenate([draw(base * z, n_in), draw(base * (~z), K - n_in)])
            print('  smart_init : %d gaussiennes dans la zone objet (%.0f %%), %d hors'
                  % (n_in, 100 * zone_frac, K - n_in))
        else:
            flat = draw(base, K)

        py, px = flat // Wi, flat % Wi
        mu = np.stack([px / (Wi - 1), py / (Hi - 1)], 1)
        t = np.clip((mu + 0.1) / 1.2, 0.01, 0.99)
        self.mu_raw.copy_(torch.tensor(np.log(t / (1 - t)), dtype=torch.float32))

        # softplus_inv(sigma - scale_min)
        s = max(sigma_xy - self.scale_min, 1e-5)
        self.logs_raw.fill_(float(np.log(np.expm1(s))))
        self.theta_raw.copy_(torch.rand(K) * math.pi)
        # profondeurs etalees mais faibles : l ordre initial est aleatoire, ce qui est le
        # bon a priori (rien ne dit qui occulte qui), et le champ peut le reorganiser.
        self.prof_raw.copy_((torch.rand(K) * 2 - 1) * 0.5)
        self.log_alpha.fill_(0.0)
        col = img[py, px, :C].clamp(0.01, 0.99)
        self.color_raw.copy_(torch.log(col / (1 - col)))
        print('  smart_init deform : %d gaussiennes, sigma_xy %.4f (%.1f px), '
              'alpha_min %.3f, scale_min %.1e'
              % (K, sigma_xy, sigma_xy * 0.5 * sum(self.img_size),
                 self.alpha_min, self.scale_min))


class SharedDeformField(nn.Module):
    """
    UN SEUL reseau pour l etat de TOUTES les gaussiennes : etat -> (K, n_out) d un coup.

    Difference avec `DeformField`, qui est evalue une fois PAR GAUSSIENNE avec sa position
    canonique en entree, donc lisse en espace par construction. Ici la position n entre pas :
    la derniere couche a une sortie par (gaussienne, parametre), soit K * n_out, et chaque
    gaussienne a ses propres poids de sortie. Rien ne lie deux gaussiennes voisines hors du
    fait qu elles lisent la meme representation cachee de l etat, de dimension `hidden`.

    `layers` compte les couches LINEAIRES, derniere comprise : layers = 2 donne
    etat -> hidden -> K * n_out.
    """

    def __init__(self, cond_dim: int, n_gaussians: int, n_out: int, hidden: int = 128,
                 layers: int = 2, pe_cond: int = 0):
        super().__init__()
        self.pe_cond = pe_cond
        self.n_gaussians, self.n_out = n_gaussians, n_out
        dims = [_fourier_dim(cond_dim, pe_cond)] + [hidden] * (layers - 1)
        self.body = nn.ModuleList(nn.Linear(i, o) for i, o in zip(dims[:-1], dims[1:]))
        self.head = nn.Linear(dims[-1], n_gaussians * n_out)
        # sortie nulle a l initialisation, meme raison que DeformField
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, cond: torch.Tensor, idx: torch.Tensor = None) -> torch.Tensor:
        """cond (B, C) -> (B, K, n_out), ou (B, M, n_out) pour les seules gaussiennes `idx`
        (seules leurs lignes de la derniere couche sont alors calculees)."""
        h = _fourier(cond, self.pe_cond)
        for lin in self.body:
            h = F.silu(lin(h))
        if idx is None:
            return self.head(h).view(h.shape[0], self.n_gaussians, self.n_out)
        rows = (idx.unsqueeze(1) * self.n_out
                + torch.arange(self.n_out, device=idx.device)).reshape(-1)
        out = F.linear(h, self.head.weight[rows], self.head.bias[rows])
        return out.view(h.shape[0], len(idx), self.n_out)


class SharedDeformSplatDecoder(DeformSplatDecoder):
    """
    `DeformSplatDecoder` dont le champ est un `SharedDeformField` : meme parametres
    canoniques, memes bornes, meme rendu avec profondeur, meme `q_sensitivity`. Seul change
    le reseau qui produit la deformation.

    `max_dmu` remplace la borne de deplacement des centres : dans le cadre FIXE de la
    camera le bras traverse une grande part de l image, et 0.25 ne suffit plus.

    `deform_alpha` : le champ sort en plus un logit d opacite par gaussienne, ajoute a
    `log_alpha` et borne a +- `max_dalpha`. En 2D l opacite DOIT dependre de l etat (q et s) :
    une partie de la scene qui passe derriere le bras, ou qui sort du champ quand la vue
    change, ne peut pas etre rendue par la seule couleur. Le plancher `alpha_min` reste, donc
    aucune gaussienne ne disparait pour de bon. Tete du champ nulle a l initialisation : le
    decodeur part exactement de l opacite constante. (Le decodeur 3D, lui, garde une opacite
    constante.)
    """

    def __init__(self, latent_dim: int, cond_dim: int, n_gaussians: int, img_size: tuple,
                 n_channels: int = 3, hidden: int = 128, layers: int = 2, pe_cond: int = 0,
                 alpha_min: float = 0.02, scale_min: float = 3.e-4,
                 depth_scale: float = 1.0, deform_color: bool = True,
                 max_dmu: float = None, deform_alpha: bool = False,
                 max_dalpha: float = 4.0):
        super().__init__(latent_dim, cond_dim, n_gaussians, img_size, n_channels,
                         hidden=hidden, layers=1, pe_pos=0, pe_cond=0,
                         alpha_min=alpha_min, scale_min=scale_min,
                         depth_scale=depth_scale, deform_color=deform_color)
        if max_dmu is not None:
            self.MAX_DMU = max_dmu
        self.deform_alpha = deform_alpha
        self.MAX_DALPHA = max_dalpha
        # canal d opacite en DERNIER : `_split` lit les canaux par indice, il l ignore
        self.field = SharedDeformField(cond_dim, n_gaussians,
                                       self.n_out + (1 if deform_alpha else 0),
                                       hidden=hidden, layers=layers, pe_cond=pe_cond)

    def _field_out(self, pos, cond, idx=None):
        return self.field(cond, idx)

    def deform(self, cond: torch.Tensor, mu0: torch.Tensor = None):
        return self._split(self.field(cond))

    def _deform_alpha(self, cond, mu0):
        out = self.field(cond)
        dla = torch.tanh(out[..., -1]) * self.MAX_DALPHA if self.deform_alpha else None
        return self._split(out), dla


# ══════════════════════════════════════════════════════════════════════════════════════
#  3D : gaussiennes METRIQUES dans le repere de la piece, deformees par un MLP partage
# ══════════════════════════════════════════════════════════════════════════════════════

def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Produit de quaternions wxyz (convention gsplat), diffusion sur les dimensions de tete."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz,
                        aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw], dim=-1)


def _axis_angle_quat(v: torch.Tensor) -> torch.Tensor:
    """Vecteur de rotation (..., 3) -> quaternion wxyz. `sqrt(|v|^2 + eps)` : le champ sort
    exactement zero a l initialisation, ou la norme n est pas derivable."""
    th = (v.pow(2).sum(-1, keepdim=True) + 1e-12).sqrt()
    return torch.cat([torch.cos(0.5 * th), v * (torch.sin(0.5 * th) / th)], dim=-1)


class SharedDeformSplat3D(nn.Module):
    """
    Equivalent 3D de `SharedDeformSplatDecoder` : etat `z = (s, q)` -> toutes les gaussiennes.

    Chaque gaussienne a des parametres CANONIQUES dans le repere metrique de la piece (centre,
    demi-axes, orientation, couleur, opacite), et UN SEUL MLP, z -> hidden -> K * 12, sort
    pour chacune un deplacement, une variation de log-demi-axes, une rotation (vecteur de
    rotation) et une variation de couleur. Derniere couche a zero : le decodeur part de sa
    scene canonique.

    Comme en 2D, aucune gaussienne ne peut disparaitre : l opacite ne depend pas de l etat et
    a un plancher (`alpha_min`), les demi-axes aussi (`scale_min`, unites de la jauge). La
    profondeur n est plus un artifice d ordre : c est la vraie geometrie, et `gsplat`
    rasterise depuis n importe quelle pose.
    """

    N_OUT = 3 + 3 + 3 + 3

    def __init__(self, cond_dim: int, n_gaussians: int, n_channels: int = 3,
                 hidden: int = 128, layers: int = 2, pe_cond: int = 0,
                 alpha_min: float = 0.02, scale_min: float = 2e-3,
                 max_dxyz: float = 2.0, max_dlogs: float = 1.0,
                 max_drot: float = math.pi, max_dcolor: float = 0.5,
                 latent_cols: tuple = ()):
        super().__init__()
        K, C = n_gaussians, n_channels
        self.n_gaussians, self.n_channels, self.cond_dim = K, C, cond_dim
        self.alpha_min, self.scale_min = alpha_min, scale_min
        self.max_dxyz, self.max_dlogs = max_dxyz, max_dlogs
        self.max_drot, self.max_dcolor = max_drot, max_dcolor
        # colonnes de l etat dont la dependance est PENALISEE (q) ; les autres (s) sont libres
        self.latent_cols = tuple(latent_cols)
        self.xyz = nn.Parameter(torch.zeros(K, 3))
        self.logs_raw = nn.Parameter(torch.zeros(K, 3))
        self.quat_raw = nn.Parameter(torch.tensor([1.0, 0, 0, 0]).repeat(K, 1))
        self.log_alpha = nn.Parameter(torch.zeros(K))
        self.color_raw = nn.Parameter(torch.zeros(K, C))
        self.field = SharedDeformField(cond_dim, K, self.N_OUT, hidden=hidden,
                                       layers=layers, pe_cond=pe_cond)

    def alpha(self):
        return self.alpha_min + (1.0 - self.alpha_min) * torch.sigmoid(self.log_alpha)

    def _split(self, out):
        dxyz = torch.tanh(out[..., 0:3]) * self.max_dxyz
        dlogs = torch.tanh(out[..., 3:6]) * self.max_dlogs
        drot = torch.tanh(out[..., 6:9]) * self.max_drot
        dcol = torch.tanh(out[..., 9:12]) * self.max_dcolor
        return dxyz, dlogs, drot, dcol

    def state(self, cond: torch.Tensor, idx: torch.Tensor = None):
        """cond (B, cond_dim) -> xyz (B,K,3), quat (B,K,4), scale (B,K,3), alpha (K,),
        color (B,K,C). `idx` restreint a un sous-ensemble de gaussiennes."""
        sl = slice(None) if idx is None else idx
        dxyz, dlogs, drot, dcol = self._split(self.field(cond, idx))
        xyz = self.xyz[sl].unsqueeze(0) + dxyz
        scale = self.scale_min + F.softplus(self.logs_raw[sl].unsqueeze(0) + dlogs)
        q0 = F.normalize(self.quat_raw[sl], dim=-1).unsqueeze(0)
        quat = F.normalize(_quat_mul(_axis_angle_quat(drot), q0), dim=-1)
        color = torch.sigmoid(self.color_raw[sl].unsqueeze(0) + dcol)
        return xyz, quat, scale, self.alpha()[sl], color

    def render(self, cond: torch.Tensor, viewmats: torch.Tensor, Ks: torch.Tensor,
               H: int, W: int) -> torch.Tensor:
        """
        cond (B, cond_dim) ; viewmats (B, V, 4, 4) monde -> camera, Ks (B, V, 3, 3).
        Les cameras peuvent differer d un etat a l autre (l iphone3 bouge) : chaque etat est
        rasterise en UN appel pour ses V vues. Retourne (B, V, C, H, W).
        """
        from gsplat import rasterization
        xyz, quat, scale, alpha, color = self.state(cond)
        out = []
        for b in range(cond.shape[0]):
            r, _, _ = rasterization(
                means=xyz[b], quats=quat[b], scales=scale[b],
                opacities=alpha.clamp(0, 0.99), colors=color[b],
                viewmats=viewmats[b], Ks=Ks[b], width=W, height=H,
                near_plane=0.01, far_plane=100.0, sh_degree=None, render_mode='RGB')
            # fond : le defaut de gsplat quand `backgrounds` vaut None est le noir
            out.append(r[..., :self.n_channels].permute(0, 3, 1, 2).clamp(0, 1))
        return torch.stack(out, 0)

    def q_sensitivity(self, cond: torch.Tensor, idx: torch.Tensor, E_ref: torch.Tensor,
                      f_px: float, h: float = 0.05):
        """
        Deplacement du contour a 1 sigma, EN PIXELS de la camera de reference (E_ref, focale
        f_px), par unite de chaque colonne `latent_cols` de l etat. Meme mesure que la
        version 2D : centre, dilatation des demi-axes et rotation, differences finies
        centrees. Le passage au pixel se fait par f / Z a la profondeur canonique de la
        gaussienne dans la camera de reference. Retourne (dep, dcol), chacun (B, M).
        """
        X = self.xyz[idx].detach()
        Zc = (X @ E_ref[:3, :3].T + E_ref[:3, 3])[:, 2].abs().clamp(min=0.3)
        px = (f_px / Zc).unsqueeze(0)                                  # (1, M)
        s0 = (self.scale_min + F.softplus(self.logs_raw[idx])).detach()
        dep2 = torch.zeros(cond.shape[0], len(idx), device=cond.device)
        dcol2 = torch.zeros_like(dep2)
        inv = 1.0 / (2 * h)
        for a in self.latent_cols:
            e = torch.zeros_like(cond)
            e[:, a] = h
            gp = self._split(self.field(cond + e, idx))
            gm = self._split(self.field(cond - e, idx))
            g_x = (gp[0] - gm[0]) * inv
            g_s = (gp[1] - gm[1]) * inv * s0.unsqueeze(0)
            g_r = (gp[2] - gm[2]) * inv * s0.max(-1).values[None, :, None]
            dep2 = dep2 + (g_x.pow(2).sum(-1) + g_s.pow(2).sum(-1)
                           + g_r.pow(2).sum(-1)) * px ** 2
            dcol2 = dcol2 + ((gp[3] - gm[3]) * inv).pow(2).sum(-1)
        return (dep2 + 1e-6).sqrt(), (dcol2 + 1e-8).sqrt()
