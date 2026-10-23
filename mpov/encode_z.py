"""
Encode the whole iphone1 crop video with the frozen encoder: z(t) = (s, q0, q1), normalized,
one row per proxy frame of iphone1. Taken from `encode_all` of the original train_dec3d_shared.py.

Then, with --s-corr, replace s at the instants refined by the original refine_worst.py
(inputs/all_corr.npz, field `ds`), as in the original run_chain_v5.sh.

    py encode_z.py --ae runs/i1_d2_frozen/ae.pt --out runs/z_all.npy --s-corr inputs/all_corr.npz
"""
import argparse
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent.resolve()


def load_encoder(ck_path, dev):
    """Frozen encoder and normalization of s, read from the autoencoder checkpoint."""
    import _bootstrap
    _bootstrap.load_config()
    import models
    from train_ae_demoxr import EncoderU
    ck = torch.load(ck_path, map_location=dev, weights_only=False)
    d_q = ck['args']['d_q']
    n_u = int(np.atleast_1d(ck['u_mean']).size)
    has_wh = any(k.startswith('whiten.') for k in ck['enc'])
    S = int(ck['args'].get('side', 256)) if 'side' in ck['args'] else 256
    enc = EncoderU(models, (S, S), d_q, n_channels=3, whiten=has_wh, n_u=n_u).to(dev)
    enc.load_state_dict(ck['enc'])
    enc.eval()
    for p_ in enc.parameters():
        p_.requires_grad_(False)
    return enc, d_q, n_u, np.atleast_1d(ck['u_mean']), np.atleast_1d(ck['u_std'])


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ae', default='runs/i1_d2_frozen/ae.pt')
    p.add_argument('--frames', default='iphone1/frames_256.npy')
    p.add_argument('--track', default='iphone1/s_track.npy')
    p.add_argument('--s-corr', default='', help='all_corr.npz of refine_worst.py (optional)')
    p.add_argument('--out', default='runs/z_all.npy')
    p.add_argument('--device', default='cuda')
    p.add_argument('--limit', type=int, default=0, help='first frames only (tests)')
    a = p.parse_args()
    dev = a.device
    s_track = np.load(HERE / a.track).astype(np.float32)
    F = np.load(HERE / a.frames, mmap_mode='r')
    T = min(len(s_track), len(F))
    if a.limit:
        T = min(T, a.limit)
    enc, d_q, n_u, u_mean, u_std = load_encoder(HERE / a.ae, dev)
    out = []
    for i in range(0, T, 256):
        k = np.arange(i, min(i + 256, T))
        img = torch.tensor(np.asarray(F[k], np.float32) / 255.0, device=dev)
        u = torch.tensor(((s_track[k].reshape(-1, n_u) - u_mean) / u_std).astype(np.float32),
                         device=dev)
        out.append(torch.cat([u, enc(img, u)], 1).cpu().numpy())
    Z = np.concatenate(out).astype(np.float32)
    print('z(t) : %d frames, dimension %d, std %s' % (T, Z.shape[1], np.round(Z.std(0), 3)))
    if a.s_corr:
        f = np.load(HERE / a.s_corr)
        for ir, ds in zip(f['i_ref'], f['ds']):
            Z[int(round(ir)), 0] += ds
        print('s corrected at %d instants' % len(f['ds']))
    out_p = HERE / a.out
    out_p.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_p, Z)
    print('->', out_p)


if __name__ == '__main__':
    main()
