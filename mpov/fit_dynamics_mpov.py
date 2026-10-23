"""
Physique LINEAIRE de la languette pour le cas mpov, meme recette que img5175/fit_dynamics.py
--model linear (celle de runs/_web_final/dyn_lin, branchee dans la demo 2D) :

    q̈ = -K q - C q̇ + G s̈ + B s + b

q = (q0, q1) vient de l'encodeur GELE de l'iphone1 (runs/<run>/z_all.npy, colonnes 1 et 2),
s de la meme table (colonne 0), l'abscisse curviligne du bras, NORMALISEE comme a
l'entrainement de l'AE. s est une ENTREE mesuree ; s̈ est le forcage inertiel de d'Alembert
(la languette vit dans le repere de la pince, c'est l'acceleration de la base qui l'excite),
B s laisse la position du bras deplacer l'equilibre.

Difference avec img5175 : l'entree y etait u = (u_x, u_y), ici elle vaut s, de dimension 1
(le mouvement du bras est a un parametre, cf. mpov/README.md). G et B sont donc d x 1.

Deux temps, comme la-bas : moindres carres sur le residu aux differences finies (solution
exacte du probleme lineaire), puis raffinement sur l'erreur de ROLLOUT (Verlet
differentiable, curriculum d'horizon), qui seul contraint la stabilite et l'amortissement.
Held-out : les --val-sec dernieres secondes. Metrique : NRMSE du rollout (1.0 = ne fait pas
mieux que la moyenne), comme img5175.

Sortie : runs/<out>/model.pt ({'model': state_dict avec K, C, G, B, b ; 'args' ; 'fps'}),
log.txt, rollout_holdout.png.

Usage :
    py fit_dynamics_mpov.py
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

HERE = Path(__file__).parent
FPS = 59.939895            # cadence REELLE de l'iphone1 (measure_drift.py), l'index de z_all
DT = 1.0 / FPS


class LinearDyn(nn.Module):
    """q̈ = -K q - C q̇ + G ü + B u + b, u de dimension n_u."""

    def __init__(self, d, n_u):
        super().__init__()
        self.K = nn.Parameter(torch.zeros(d, d))
        self.C = nn.Parameter(torch.zeros(d, d))
        self.G = nn.Parameter(torch.zeros(d, n_u))
        self.B = nn.Parameter(torch.zeros(d, n_u))
        self.b = nn.Parameter(torch.zeros(d))

    def accel(self, q, qd, u, udd):
        return -q @ self.K.T - qd @ self.C.T + udd @ self.G.T + u @ self.B.T + self.b


def rollout(model, q0, v0, u, udd, dt):
    """Verlet semi-implicite, le meme que img5175/fit_dynamics.rollout."""
    q, v = q0, v0
    out = [q]
    for k in range(u.shape[1] - 1):
        v = v + dt * model.accel(q, v, u[:, k], udd[:, k])
        q = q + dt * v
        out.append(q)
    return torch.stack(out, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--z', default='runs/dec3d_shared_p1/z_all.npy')
    ap.add_argument('--val-sec', type=float, default=60.0)
    ap.add_argument('--horizon', type=float, default=2.0)
    ap.add_argument('--sg-win', type=int, default=9)
    ap.add_argument('--sg-win-v0', type=int, default=5)
    ap.add_argument('--smooth-q', type=float, default=1.0)
    ap.add_argument('--roll-steps', type=int, default=1000)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--out', default='dyn_lin')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    out = HERE / 'runs' / a.out
    out.mkdir(parents=True, exist_ok=True)

    Z = np.load(HERE / a.z).astype(np.float64)
    u, q = Z[:, :1], Z[:, 1:]
    T, d = q.shape
    q_raw = q.copy()
    if a.smooth_q > 0:
        q = gaussian_filter1d(q, a.smooth_q, axis=0, mode='nearest')
    w = a.sg_win | 1
    qd = savgol_filter(q, w, 3, deriv=1, delta=DT, axis=0)
    qd0 = savgol_filter(q, a.sg_win_v0 | 1, 2, deriv=1, delta=DT, axis=0)
    qdd = savgol_filter(q, w, 3, deriv=2, delta=DT, axis=0)
    udd = savgol_filter(u, w, 3, deriv=2, delta=DT, axis=0)
    print('T = %d, q std %s, s std %.2f, s'' std %.1f'
          % (T, np.round(q.std(0), 3), u.std(), udd.std()))

    n_val = int(a.val_sec * FPS)
    T_tr = T - n_val
    H = int(a.horizon * FPS)
    tq, tqd0, tu, tudd = (torch.tensor(x, dtype=torch.float32, device=dev)
                          for x in (q, qd0, u, udd))

    model = LinearDyn(d, 1).to(dev)
    X = np.concatenate([q, qd, udd, u, np.ones((T, 1))], 1)[:T_tr]
    sol, *_ = np.linalg.lstsq(X, qdd[:T_tr], rcond=None)
    sol = torch.tensor(sol.T, dtype=torch.float32, device=dev)
    with torch.no_grad():
        model.K.copy_(-sol[:, :d]); model.C.copy_(-sol[:, d:2 * d])
        model.G.copy_(sol[:, 2 * d:2 * d + 1]); model.B.copy_(sol[:, 2 * d + 1:2 * d + 2])
        model.b.copy_(sol[:, -1])
    r = qdd[:T_tr] - X @ sol.T.cpu().numpy()
    log = open(out / 'log.txt', 'w')

    def say(m):
        print(m, flush=True)
        log.write(m + '\n')
        log.flush()

    say('moindres carres : residu FD relatif %s' % np.round(r.std(0) / qdd[:T_tr].std(0), 3))

    rng = np.random.default_rng(0)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr * 0.3)
    t0 = time.time()
    for it in range(1, a.roll_steps + 1):
        hh = min(H, int(H * min(1.0, 0.2 + 0.8 * it / (0.5 * a.roll_steps))))
        starts = rng.integers(w, T_tr - hh, size=a.batch)
        idx = torch.tensor(np.stack([np.arange(s, s + hh) for s in starts]), device=dev)
        sim = rollout(model, tq[idx[:, 0]], tqd0[idx[:, 0]], tu[idx], tudd[idx], DT)
        if not torch.isfinite(sim).all():
            opt.zero_grad(set_to_none=True)
            continue
        loss = torch.nn.functional.huber_loss(sim, tq[idx], delta=1.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if it % 200 == 0:
            say('rollout %5d  h=%.1fs  huber %.4f  (%.0f s)'
                % (it, hh * DT, loss.item(), time.time() - t0))

    @torch.no_grad()
    def nrmse(lo, hi, Hh, n=60):
        starts = np.linspace(lo, hi - Hh - 1, n).astype(int)
        idx = torch.tensor(np.stack([np.arange(s, s + Hh) for s in starts]), device=dev)
        sim = rollout(model, tq[idx[:, 0]], tqd0[idx[:, 0]], tu[idx], tudd[idx], DT)
        e = (sim - tq[idx]).pow(2).mean((0, 1)).sqrt()
        return (e / tq[lo:hi].std(0)).cpu().numpy(), sim, tq[idx], starts

    tr, _, _, _ = nrmse(w, T_tr, H)
    va, sim, qs, starts = nrmse(T_tr, T, H)
    say('NRMSE rollout %.1f s : train %s (global %.3f) | held-out %s (global %.3f)'
        % (a.horizon, np.round(tr, 3), tr.mean(), np.round(va, 3), va.mean()))
    floor = (q_raw - gaussian_filter1d(q_raw, max(a.smooth_q, 1e-6), axis=0,
                                       mode='nearest')).std(0) / q_raw.std(0)
    say('plancher de bruit de q (lissage sigma=%g) : %s' % (a.smooth_q, np.round(floor, 3)))
    for hs in (0.25, 0.5, 1.0, a.horizon):
        v, _, _, _ = nrmse(T_tr, T, int(hs * FPS))
        say('  held-out a %.2f s : %s (global %.3f)' % (hs, np.round(v, 3), v.mean()))
    Kn = model.K.detach().cpu().numpy()
    f_hz = np.sqrt(np.abs(np.linalg.eigvals(Kn).real)) / (2 * np.pi)
    say('frequences propres %s Hz' % np.round(f_hz, 2))
    torch.save({'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                'args': vars(a), 'fps': FPS}, out / 'model.pt')

    k = [0, len(starts) // 2, len(starts) - 1]
    fig, ax = plt.subplots(d, len(k), figsize=(5 * len(k), 2.2 * d), sharex=True)
    tt = np.arange(qs.shape[1]) * DT
    for c, j in enumerate(k):
        for r_ in range(d):
            ax[r_, c].plot(tt, qs[j, :, r_].cpu(), 'k-', lw=1, label='mesure')
            ax[r_, c].plot(tt, sim[j, :, r_].cpu(), 'r-', lw=1, label='rollout')
            ax[r_, c].set_ylabel('q%d' % r_, fontsize=8)
        ax[0, c].set_title('held-out, depart t=%.0f s' % (starts[j] / FPS), fontsize=9)
    ax[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'rollout_holdout.png', dpi=100)
    log.close()
    print('ecrit', out)


if __name__ == '__main__':
    main()
