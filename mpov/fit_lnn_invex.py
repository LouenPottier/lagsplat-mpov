"""
LNN du cas mpov : masse CONSTANTE, dissipation CONSTANTE, potentiel INVEXE.

    M q'' = -grad V(q) - C q' + G s''

  M = L L^T + eps I                     constante, SDP
  C = Lc Lc^T                           constante, SDP (dissipation de Rayleigh q'^T C q' / 2)
  V(q) = g(Phi(q) - Phi(q_r))           invexe : un seul point stationnaire, le minimum q_r
    Phi = (I + r_K) o ... o (I + r_1)   difféomorphisme i-ResNet (Behrmann 2019) : Lip(r_k) < 1
                                        par normes spectrales EXACTES (2 x hidden, sans
                                        iterations de puissance, compatible double derivee)
    g(y) = y^T A y / 2 + h(y) - h(0) - grad h(0) . y
                                        A SDP, h ICNN (convexe) : g strictement convexe,
                                        minimum unique en y = 0 (correction de Bregman)
  G s''                                 forcage de d'Alembert (le bras accelere la pince)

Meme protocole que fit_dynamics_mpov.py (modele lineaire de reference, runs/dyn_lin) : memes
q(t) et s(t), meme lissage, held-out = dernieres --val-sec secondes, perte de ROLLOUT (Huber,
curriculum d'horizon), meme integrateur (Euler semi-implicite a dt = 1/fps, --sub sous-pas),
meme NRMSE. Depart : le modele lineaire (A = K symetrise, C = C symetrise rendu SDP, G, Phi = Id,
h ~ 0), donc l'apprentissage part d'un oscillateur deja ajuste.

Sorties runs/<out>/ : model.pt, log.txt, rollout_holdout.png, potential.png (V(q) et nuage).

    py fit_lnn_invex.py --out lnn_invex
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

HERE = Path(__file__).parent
FPS = 59.939895
DT = 1.0 / FPS


class LipLinear(nn.Module):
    """Lineaire de norme spectrale <= 1 : W / max(1, sigma_max(W)) (exacte, petite matrice)."""

    def __init__(self, i, o):
        super().__init__()
        self.lin = nn.Linear(i, o)

    def forward(self, x):
        W = self.lin.weight
        s = torch.linalg.matrix_norm(W, ord=2)
        return F.linear(x, W / torch.clamp(s, min=1.0), self.lin.bias)


class Diffeo(nn.Module):
    def __init__(self, d, hidden, n_blocks, coeff=0.9):
        super().__init__()
        self.coeff = coeff
        self.blocks = nn.ModuleList([nn.Sequential(LipLinear(d, hidden), nn.Tanh(),
                                                   LipLinear(hidden, d))
                                     for _ in range(n_blocks)])
        self.raw = nn.Parameter(torch.zeros(n_blocks))          # Phi = Id au depart

    def forward(self, q):
        for b, r in zip(self.blocks, self.raw):
            q = q + self.coeff * torch.tanh(r) * b(q)
        return q


class ICNN(nn.Module):
    """h(y) convexe : softplus(W0 y + b0) puis couches a poids positifs."""

    def __init__(self, d, hidden, layers=2):
        super().__init__()
        self.Wy = nn.ModuleList([nn.Linear(d, hidden) for _ in range(layers)])
        self.Wz = nn.ParameterList([nn.Parameter(torch.randn(hidden, hidden) * 0.1)
                                    for _ in range(layers - 1)])
        self.out = nn.Parameter(torch.randn(hidden) * 0.01)

    def forward(self, y):
        z = F.softplus(self.Wy[0](y))
        for k, Wz in enumerate(self.Wz):
            z = F.softplus(z @ F.softplus(Wz).T + self.Wy[k + 1](y))
        return z @ F.softplus(self.out)


class LNN(nn.Module):
    def __init__(self, d=2, n_u=1, hidden=32, blocks=2, icnn_hidden=32, eps_m=1e-3):
        super().__init__()
        self.d, self.eps_m = d, eps_m
        self.Lm = nn.Parameter(torch.eye(d))
        self.Lc = nn.Parameter(torch.eye(d) * 0.5)
        self.La = nn.Parameter(torch.eye(d))
        self.G = nn.Parameter(torch.zeros(d, n_u))
        self.q_r = nn.Parameter(torch.zeros(d))
        self.phi = Diffeo(d, hidden, blocks)
        self.h = ICNN(d, icnn_hidden)

    def mass(self):
        L = torch.tril(self.Lm)
        return L @ L.T + self.eps_m * torch.eye(self.d, device=L.device)

    def damping(self):
        L = torch.tril(self.Lc)
        return L @ L.T

    def A(self):
        L = torch.tril(self.La)
        return L @ L.T

    def V(self, q):
        y = self.phi(q) - self.phi(self.q_r[None])
        y0 = torch.zeros(1, self.d, device=q.device, requires_grad=True)
        with torch.enable_grad():
            h0 = self.h(y0)
            gh0, = torch.autograd.grad(h0.sum(), y0, create_graph=True)
        quad = 0.5 * ((y @ self.A()) * y).sum(-1)
        return quad + self.h(y) - h0 - (y * gh0).sum(-1)

    def grad_V(self, q, create_graph=True):
        with torch.enable_grad():
            q = q if q.requires_grad else q.requires_grad_(True)
            g, = torch.autograd.grad(self.V(q).sum(), q, create_graph=create_graph)
        return g

    def accel(self, q, v, udd, create_graph=True):
        f = -self.grad_V(q, create_graph) - v @ self.damping().T + udd @ self.G.T
        return torch.linalg.solve(self.mass(), f.T).T


def rollout(model, q0, v0, udd, dt, sub=1, create_graph=True):
    q, v = q0, v0
    out = [q]
    h = dt / sub
    for k in range(udd.shape[1] - 1):
        for _ in range(sub):
            v = v + h * model.accel(q, v, udd[:, k], create_graph)
            q = q + h * v
        if not create_graph:
            q, v = q.detach(), v.detach()
        out.append(q)
    return torch.stack(out, 1)


def init_from_linear(model, path, dev):
    ck = torch.load(path, weights_only=False)['model']
    K = ck['K'].double()
    C = ck['C'].double()
    Ks = 0.5 * (K + K.T)
    w, U = torch.linalg.eigh(Ks)
    Ks = U @ torch.diag(w.clamp(min=1.0)) @ U.T
    Cs = 0.5 * (C + C.T)
    w, U = torch.linalg.eigh(Cs)
    Cs = U @ torch.diag(w.clamp(min=1e-2)) @ U.T
    with torch.no_grad():
        model.La.copy_(torch.linalg.cholesky(Ks).float())
        model.Lc.copy_(torch.linalg.cholesky(Cs).float())
        model.G.copy_(ck['G'].float())
    return Ks, Cs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--z', default='runs/dec3d_shared_p1/z_all.npy')
    ap.add_argument('--init-linear', default='runs/dyn_lin/model.pt')
    ap.add_argument('--val-sec', type=float, default=60.0)
    ap.add_argument('--horizon', type=float, default=2.0)
    ap.add_argument('--sg-win', type=int, default=9)
    ap.add_argument('--sg-win-v0', type=int, default=5)
    ap.add_argument('--smooth-q', type=float, default=1.0)
    ap.add_argument('--steps', type=int, default=1500)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--sub', type=int, default=1, help='sous-pas d\'integration par frame')
    ap.add_argument('--hidden', type=int, default=32)
    ap.add_argument('--blocks', type=int, default=2)
    ap.add_argument('--out', default='lnn_invex')
    a = ap.parse_args()
    torch.manual_seed(0)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    out = HERE / 'runs' / a.out
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / 'log.txt', 'w', encoding='utf-8')

    def say(m):
        print(m, flush=True)
        log.write(m + '\n')
        log.flush()

    Z = np.load(HERE / a.z).astype(np.float64)
    u, q = Z[:, :1], Z[:, 1:]
    T, d = q.shape
    q_raw = q.copy()
    q = gaussian_filter1d(q, a.smooth_q, axis=0, mode='nearest')
    qd0 = savgol_filter(q, a.sg_win_v0 | 1, 2, deriv=1, delta=DT, axis=0)
    udd = savgol_filter(u, a.sg_win | 1, 3, deriv=2, delta=DT, axis=0)
    T_tr = T - int(a.val_sec * FPS)
    H = int(a.horizon * FPS)
    tq, tqd0, tudd = (torch.tensor(x, dtype=torch.float32, device=dev) for x in (q, qd0, udd))

    model = LNN(d, 1, a.hidden, a.blocks).to(dev)
    init_from_linear(model, HERE / a.init_linear, dev)
    n_par = sum(p.numel() for p in model.parameters())
    say('%s : %d parametres, T = %d (entrainement %d), depart = modele lineaire %s'
        % (a.out, n_par, T, T_tr, a.init_linear))

    def freqs():
        with torch.no_grad():
            qr = model.q_r.detach()[None].clone().requires_grad_(True)
        with torch.enable_grad():
            g = model.grad_V(qr, True)
            Hm = torch.stack([torch.autograd.grad(g[0, i], qr, retain_graph=True)[0][0]
                              for i in range(d)])
        ev = torch.linalg.eigvals(torch.linalg.solve(model.mass().detach(), Hm.detach())).real
        return np.sqrt(np.abs(ev.cpu().numpy())) / (2 * np.pi)

    say('depart : frequences au repos %s Hz' % np.round(freqs(), 2))
    w = a.sg_win | 1
    rng = np.random.default_rng(0)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps, eta_min=a.lr * 0.05)
    t0 = time.time()
    for it in range(1, a.steps + 1):
        hh = min(H, int(H * min(1.0, 0.2 + 0.8 * it / (0.5 * a.steps))))
        starts = rng.integers(w, T_tr - hh, size=a.batch)
        idx = torch.tensor(np.stack([np.arange(s, s + hh) for s in starts]), device=dev)
        sim = rollout(model, tq[idx[:, 0]], tqd0[idx[:, 0]], tudd[idx], DT, a.sub)
        if not torch.isfinite(sim).all():
            opt.zero_grad(set_to_none=True)
            say('it %d : rollout non fini, lot saute' % it)
            continue
        loss = F.huber_loss(sim, tq[idx], delta=1.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        if it % 100 == 0 or it == 1:
            say('it %5d  h=%.1fs  huber %.4f  phi %s  (%.0f s)'
                % (it, hh * DT, loss.item(),
                   np.round((model.phi.coeff * torch.tanh(model.phi.raw)).detach().cpu().numpy(), 3),
                   time.time() - t0))
        if it % 100 == 0:                                       # checkpoint intermediaire
            torch.save({'model': model.state_dict(), 'args': vars(a), 'fps': FPS, 'it': it},
                       out / 'model.pt')
    torch.save({'model': model.state_dict(), 'args': vars(a), 'fps': FPS}, out / 'model.pt')

    def nrmse(lo, hi, Hh, n=60):
        starts = np.linspace(lo, hi - Hh - 1, n).astype(int)
        idx = torch.tensor(np.stack([np.arange(s, s + Hh) for s in starts]), device=dev)
        sim = rollout(model, tq[idx[:, 0]], tqd0[idx[:, 0]], tudd[idx], DT, a.sub,
                      create_graph=False)
        e = (sim - tq[idx]).pow(2).mean((0, 1)).sqrt()
        return (e / tq[lo:hi].std(0)).detach().cpu().numpy(), sim.detach(), tq[idx], starts

    tr, _, _, _ = nrmse(w, T_tr, H)
    say('NRMSE rollout %.1f s : train %s (global %.3f)' % (a.horizon, np.round(tr, 3), tr.mean()))
    res = {}
    for hs in (0.25, 0.5, 1.0, a.horizon):
        v_, sim, qs, starts = nrmse(T_tr, T, int(hs * FPS))
        res[hs] = v_.tolist()
        say('  held-out a %.2f s : %s (global %.3f)' % (hs, np.round(v_, 3), v_.mean()))
    floor = (q_raw - q).std(0) / q_raw.std(0)
    say('plancher de bruit de q (lissage sigma=%g) : %s' % (a.smooth_q, np.round(floor, 3)))
    say('fin : frequences au repos %s Hz, q_r = %s, M = %s, C = %s'
        % (np.round(freqs(), 2), np.round(model.q_r.detach().cpu().numpy(), 3),
           np.round(model.mass().detach().cpu().numpy(), 3).tolist(),
           np.round(model.damping().detach().cpu().numpy(), 3).tolist()))
    (out / 'eval.json').write_text(json.dumps({'nrmse_holdout': res}, indent=1))

    # rollouts held-out
    k = [0, len(starts) // 2, len(starts) - 1]
    fig, ax = plt.subplots(d, len(k), figsize=(15, 4.4), sharex=True)
    tt = np.arange(qs.shape[1]) * DT
    for c, j in enumerate(k):
        for r_ in range(d):
            ax[r_, c].plot(tt, qs[j, :, r_].cpu(), 'k-', lw=1, label='mesure')
            ax[r_, c].plot(tt, sim[j, :, r_].cpu(), 'r-', lw=1, label='LNN invexe')
            ax[r_, c].set_ylabel('q%d (encodeur)' % r_, fontsize=8)
            ax[r_, c].grid(True, alpha=0.3)
        ax[0, c].set_title('held-out, depart t=%.0f s' % (starts[j] / FPS), fontsize=9)
    ax[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'rollout_holdout.png', dpi=100)

    # potentiel : V(q) appris contre la forme quadratique du modele lineaire
    g0 = np.linspace(q_raw[:, 0].min() - 1, q_raw[:, 0].max() + 1, 120)
    g1 = np.linspace(q_raw[:, 1].min() - 1, q_raw[:, 1].max() + 1, 120)
    Q = torch.tensor(np.stack(np.meshgrid(g0, g1, indexing='ij'), -1).reshape(-1, 2),
                     dtype=torch.float32, device=dev)
    with torch.no_grad():
        Vq = model.V(Q).reshape(120, 120).cpu().numpy()
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    im = ax.contourf(g0, g1, Vq.T, levels=30, cmap='viridis')
    ax.contour(g0, g1, Vq.T, levels=15, colors='w', linewidths=0.4)
    ax.scatter(q_raw[::25, 0], q_raw[::25, 1], s=1, c='r', alpha=0.3)
    qr = model.q_r.detach().cpu().numpy()
    ax.plot(qr[0], qr[1], 'w*', ms=12)
    plt.colorbar(im, ax=ax)
    ax.set_xlabel('q0 (encodeur)'); ax.set_ylabel('q1 (encodeur)')
    ax.set_title('potentiel invexe appris V(q), nuage des donnees, minimum (etoile)')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / 'potential.png', dpi=100)
    say('ecrit %s' % out)


if __name__ == '__main__':
    main()
