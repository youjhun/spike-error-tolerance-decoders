# %% [markdown]
# # Spike-error tolerance of rate-based motor decoders — v3 (clean rerun)
#
# **What this notebook does (one run, ~20–40 min on a Colab T4 with `RUN_DEEP=True`; ~10 min without):**
#
# | Exp | Question | Output |
# |---|---|---|
# | E0 | Data statistics (sparsity, firing rate, active neurons per bin) | `results.json` |
# | E1 | Clean decoding accuracy, trial-level 5-fold CV | `fig0_baseline.png` |
# | E2 | Retention ρ vs SER under **two error definitions** (per-spike, as in the v2 report; per-1-ms-bin, our reading of Lee et al. 2024) | `fig1_ser_definitions.png` |
# | E3 | Does temporal integration (bin width / smoothing) explain the tolerance? | `fig2_integration.png` |
# | E4 | Does population size explain the tolerance? | `fig3_population.png` |
# | E5 | Population redundancy (PCA participation ratio) | `results.json` |
#
# Everything ends up in `results/` (figures + `results.json` + `macros.tex`). Download `results.zip` at the end
# and drop it next to `report_v3.tex`; the report reads every number from `macros.tex`.
#
# Fixes relative to v2: no dynamic weighting (α), no prediction smoothing in the main analysis, smoothing applied
# per trial when used, corruption repeated (`N_CORRUPT_REPEATS`) instead of a single draw, bootstrap CIs and
# Benjamini–Hochberg correction, and both SER definitions reported side by side.

# %%
# ── Setup ───────────────────────────────────────────────────────────────
# In Colab: mount Drive so the .mat file can be read.
import os, sys, json, time, math, zipfile
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')

import numpy as np
import scipy.io as sio
from scipy.ndimage import gaussian_filter1d
from scipy import stats
import matplotlib
matplotlib.use('Agg') if not IN_COLAB else None
import matplotlib.pyplot as plt

# %%
# ── Configuration (edit MAT_PATH first) ─────────────────────────────────
CONFIG = dict(
    MAT_PATH = '/content/drive/MyDrive/Data/data_and_scripts/source_data/processed/MM_S1_processed.mat',
    OUT_DIR = 'results',
    BIN_MS = 10,                 # bin width of the processed pmd-1 file
    N_LAGS = 20,                 # 200 ms history at 10 ms bins
    N_FOLDS = 5,
    SEED = 0,
    SER_LEVELS = [0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
    N_CORRUPT_REPEATS = 5,       # independent corruption draws per (fold, SER)
    RIDGE_ALPHAS = [1, 10, 100, 1000, 10000],
    RUN_MLP = True,
    RUN_DEEP = True,             # LSTM + causal Transformer (use a GPU runtime)
    DEEP_EPOCHS = 30,
    SMOOTH_SIGMA_SENS = 3,       # sensitivity analysis only (bins)
    INTEGRATION_P = [1e-3, 1e-2],
    POP_SIZES = [20, 40, 80, 161],
    POP_SUBSETS = 5,
    QUICK = False,               # True = tiny run for debugging
)

# Values from Lee et al., Nat. Electron. 7, 313–324 (2024), p.7 — reference only (different data, SNN, 1 ms SER)
PAPER_REF = {'r_neural_M1': 0.9324, 'retention_at_1e-3_M1': 0.888, 'retention_at_1e-3_S1': 0.854}

# %%
# ── Data loading ────────────────────────────────────────────────────────
def _orient(a, T):
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        a = a[:, None]
    if a.shape[0] != T and a.shape[1] == T:
        a = a.T
    return a

def load_pmd1(path):
    """Returns list of (S, V): S = spike counts (T x N, M1 then PMd), V = hand velocity (T x 2, cm/s)."""
    mat = sio.loadmat(path, struct_as_record=False, squeeze_me=True)
    D = mat['Data']
    trials = []
    n_trials = len(D.kinematics)
    for i in range(n_trials):
        kin = np.asarray(D.kinematics[i], dtype=float)
        T = kin.shape[0]
        m1 = _orient(D.neural_data_M1[i], T)
        pmd = _orient(D.neural_data_PMd[i], T)
        S = np.hstack([m1, pmd])
        V = kin[:, 2:4]                       # columns 2,3 = vx, vy (as in the v2 notebook)
        T = min(len(S), len(V))
        trials.append((S[:T], V[:T]))
    n_m1 = _orient(D.neural_data_M1[0], np.asarray(D.kinematics[0]).shape[0]).shape[1]
    return trials, n_m1

# %%
# ── Spike-error models ──────────────────────────────────────────────────
def corrupt_per_spike(S, p, rng):
    """v2 report model: SER = (N_miss + N_false) / N_spike, with p_miss = p_false = p/2.
    Each spike is deleted w.p. p/2; Binomial(N_spike, p/2) false spikes are placed uniformly at random."""
    if p == 0:
        return S.copy()
    S = S.astype(np.int64)
    out = S - rng.binomial(S, p / 2)
    n_false = rng.binomial(int(S.sum()), p / 2)
    if n_false:
        flat = out.reshape(-1)
        np.add.at(flat, rng.integers(0, flat.size, n_false), 1)
    return out.astype(float)

def corrupt_per_1ms(S, p, rng, sub=10):
    """Our reading of Lee et al. 2024: each 1 ms slot of each channel is in error w.p. p
    (a spike in it is missed, or an empty slot gets a false detection). A 10 ms bin = 10 slots."""
    if p == 0:
        return S.copy()
    S = S.astype(np.int64)
    c = np.minimum(S, sub)
    out = S - rng.binomial(c, p) + rng.binomial(sub - c, p)
    return out.astype(float)

CORRUPTORS = {'per_spike': corrupt_per_spike, 'per_1ms': corrupt_per_1ms}

# %%
# ── Features ────────────────────────────────────────────────────────────
def lag_trial(S, V, L):
    """Causal lag window built inside one trial: x_t = [s_{t-L+1}, ..., s_t]."""
    T, N = S.shape
    if T < L:
        return np.zeros((0, L, N), np.float32), np.zeros((0, V.shape[1]), np.float32)
    idx = np.arange(L)[None, :] + np.arange(T - L + 1)[:, None]
    return S[idx].astype(np.float32), V[L - 1:].astype(np.float32)

def assemble(trials, L, corrupt=None, p=0.0, rng=None, rebin=1, neurons=None):
    Xs, Ys, lens = [], [], []
    for S, V in trials:
        if neurons is not None:
            S = S[:, neurons]
        if corrupt is not None and p > 0:
            S = corrupt(S, p, rng)
        if rebin > 1:
            T = (len(S) // rebin) * rebin
            S = S[:T].reshape(-1, rebin, S.shape[1]).sum(1)
            V = V[:T].reshape(-1, rebin, V.shape[1]).mean(1)
        X, Y = lag_trial(S, V, L)
        Xs.append(X); Ys.append(Y); lens.append(len(Y))
    return np.concatenate(Xs), np.concatenate(Ys), np.array(lens)

def pearson_cols(Y, P):
    return np.array([stats.pearsonr(Y[:, j], P[:, j])[0] for j in range(Y.shape[1])])

def smooth_per_trial(P, lens, sigma):
    if not sigma:
        return P
    out, s = np.empty_like(P), 0
    for n in lens:
        out[s:s + n] = gaussian_filter1d(P[s:s + n], sigma, axis=0) if n > 1 else P[s:s + n]
        s += n
    return out

# %%
# ── Decoders ────────────────────────────────────────────────────────────
class RidgeDecoder:
    """Ridge with standardised inputs; alpha chosen by inner trial-level CV using one eigendecomposition."""
    name = 'Ridge'
    def __init__(self, alphas):
        self.alphas = alphas
    @staticmethod
    def _fit_path(X, Y, alphas):
        mu, sd = X.mean(0), X.std(0) + 1e-8
        Z = (X - mu) / sd
        ym = Y.mean(0)
        G = Z.T.astype(np.float64) @ Z
        w, Q = np.linalg.eigh(G)
        B = Q.T @ (Z.T.astype(np.float64) @ (Y - ym))
        Ws = [Q @ (B / (w + a)[:, None]) for a in alphas]
        return mu, sd, ym, Ws
    def fit(self, X, Y, lens, rng):
        X = X.reshape(len(X), -1)
        # inner 3-fold by trial
        tid = np.repeat(np.arange(len(lens)), lens)
        perm = rng.permutation(len(lens)); folds = np.array_split(perm, 3)
        scores = np.zeros(len(self.alphas))
        for f in folds:
            va = np.isin(tid, f)
            mu, sd, ym, Ws = self._fit_path(X[~va], Y[~va], self.alphas)
            Zv = (X[va] - mu) / sd
            for k, W in enumerate(Ws):
                scores[k] += pearson_cols(Y[va], Zv @ W + ym).mean()
        self.alpha = self.alphas[int(np.argmax(scores))]
        self.mu, self.sd, self.ym, Ws = self._fit_path(X, Y, [self.alpha])
        self.W = Ws[0]
        return self
    def predict(self, X):
        return ((X.reshape(len(X), -1) - self.mu) / self.sd) @ self.W + self.ym

try:
    import torch, torch.nn as nn
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
except ImportError:
    torch = None

if torch is not None:
    class _MLP(nn.Module):
        def __init__(self, L, N, out=2):
            super().__init__()
            self.net = nn.Sequential(nn.Flatten(), nn.Linear(L * N, 512), nn.ReLU(), nn.Dropout(0.2),
                                     nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, out))
        def forward(self, x):
            return self.net(x)

    class _LSTM(nn.Module):
        def __init__(self, L, N, out=2, h=128):
            super().__init__()
            self.lstm = nn.LSTM(N, h, num_layers=2, batch_first=True, dropout=0.2)
            self.head = nn.Linear(h, out)
        def forward(self, x):
            y, _ = self.lstm(x)
            return self.head(y[:, -1])

    class _CausalTransformer(nn.Module):
        def __init__(self, L, N, out=2, d=128, heads=4, layers=3):
            super().__init__()
            self.inp = nn.Linear(N, d)
            self.pos = nn.Parameter(torch.zeros(1, L, d))
            layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=256, dropout=0.1, batch_first=True)
            self.enc = nn.TransformerEncoder(layer, layers)
            self.head = nn.Linear(d, out)
            self.register_buffer('mask', torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1))
        def forward(self, x):
            z = self.enc(self.inp(x) + self.pos, mask=self.mask)
            return self.head(z[:, -1])

    class TorchDecoder:
        def __init__(self, name, cls, epochs):
            self.name, self.cls, self.epochs = name, cls, epochs
        def fit(self, X, Y, lens, rng):
            torch.manual_seed(int(rng.integers(1 << 30)))
            self.mu = X.reshape(-1, X.shape[-1]).mean(0); self.sd = X.reshape(-1, X.shape[-1]).std(0) + 1e-6
            self.ym, self.ys = Y.mean(0), Y.std(0) + 1e-6
            tid = np.repeat(np.arange(len(lens)), lens)
            va_trials = rng.choice(len(lens), max(1, len(lens) // 10), replace=False)
            va = np.isin(tid, va_trials)
            Xt = torch.tensor((X - self.mu) / self.sd, dtype=torch.float32)
            Yt = torch.tensor((Y - self.ym) / self.ys, dtype=torch.float32)
            tr_idx = np.where(~va)[0]; va_idx = np.where(va)[0]
            L, N = X.shape[1], X.shape[2]
            self.model = self.cls(L, N).to(DEVICE)
            opt = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
            best, best_state, patience = -np.inf, None, 0
            for ep in range(self.epochs):
                self.model.train()
                for b in np.array_split(rng.permutation(tr_idx), max(1, len(tr_idx) // 512)):
                    xb, yb = Xt[b].to(DEVICE), Yt[b].to(DEVICE)
                    opt.zero_grad()
                    loss = ((self.model(xb) - yb) ** 2).mean()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    opt.step()
                pv = self._raw(Xt[va_idx])
                sc = pearson_cols(Y[va_idx], pv * self.ys + self.ym).mean()
                if sc > best:
                    best, patience = sc, 0
                    best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience += 1
                    if patience >= 5:
                        break
            self.model.load_state_dict(best_state)
            return self
        def _raw(self, Xt):
            self.model.eval(); out = []
            with torch.no_grad():
                for i in range(0, len(Xt), 4096):
                    out.append(self.model(Xt[i:i + 4096].to(DEVICE)).cpu().numpy())
            return np.concatenate(out)
        def predict(self, X):
            Xt = torch.tensor((X - self.mu) / self.sd, dtype=torch.float32)
            return self._raw(Xt) * self.ys + self.ym

def make_decoders(cfg):
    decs = [lambda: RidgeDecoder(cfg['RIDGE_ALPHAS'])]
    if torch is not None and cfg['RUN_MLP']:
        decs.append(lambda: TorchDecoder('MLP', _MLP, cfg['DEEP_EPOCHS']))
    if torch is not None and cfg['RUN_DEEP']:
        decs.append(lambda: TorchDecoder('LSTM', _LSTM, cfg['DEEP_EPOCHS']))
        decs.append(lambda: TorchDecoder('Transformer', _CausalTransformer, cfg['DEEP_EPOCHS']))
    return decs

# %%
# ── Statistics helpers ──────────────────────────────────────────────────
def boot_ci(vals, rng, n=2000):
    vals = np.asarray(vals, float)
    if len(vals) < 2:
        return float(vals.mean()), float('nan'), float('nan')
    bs = rng.choice(vals, (n, len(vals)), replace=True).mean(1)
    return float(vals.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

def bh(pvals, q=0.05):
    p = np.asarray(pvals, float); m = len(p)
    order = np.argsort(p); ranked = p[order] * m / (np.arange(m) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m); out[order] = np.minimum(adj, 1)
    return out, out < q

def fisher_paired_p(r_clean, r_ser):
    """Paired t-test on Fisher-z transformed per-fold correlations (as in Lee et al. 2024)."""
    z0, z1 = np.arctanh(np.clip(r_clean, -0.999999, 0.999999)), np.arctanh(np.clip(r_ser, -0.999999, 0.999999))
    if np.allclose(z0, z1):
        return 1.0
    return float(stats.ttest_rel(z0, z1).pvalue)

def pkey(p):
    if p == 0:
        return '0'
    m, e = f'{p:.0e}'.split('e')
    return f'{m}e{int(e)}'

# %%
# ── Experiments ─────────────────────────────────────────────────────────
def run_all(cfg):
    t0 = time.time()
    rng = np.random.default_rng(cfg['SEED'])
    os.makedirs(cfg['OUT_DIR'], exist_ok=True)
    trials, n_m1 = load_pmd1(cfg['MAT_PATH'])
    if cfg['QUICK']:
        trials = trials[:60]
    R = {'config': {k: v for k, v in cfg.items()}, 'paper_ref': PAPER_REF}
    L = cfg['N_LAGS']

    # E0 — data statistics
    S_all = np.vstack([S for S, _ in trials])
    N = S_all.shape[1]
    R['data'] = dict(n_trials=len(trials), n_neurons=N, n_m1=int(n_m1), n_pmd=int(N - n_m1),
                     n_bins=int(len(S_all)), sparsity=float((S_all == 0).mean()),
                     mean_rate_hz=float(S_all.mean() / (cfg['BIN_MS'] / 1000)),
                     kbar=float((S_all > 0).sum(1).mean()), max_count=int(S_all.max()))
    # added false-spike rate implied by each definition (Hz per neuron), analytic
    rate = R['data']['mean_rate_hz']
    R['added_false_hz'] = {pkey(p): {'per_spike': rate * p / 2, 'per_1ms': 1000 * p * (1 - rate / 1000)}
                           for p in cfg['SER_LEVELS'] if p > 0}
    print('E0', R['data'])

    folds = np.array_split(rng.permutation(len(trials)), cfg['N_FOLDS'])

    # E1 + E2 — clean accuracy and SER sweep for every decoder, both definitions
    R['e1'], R['e2'] = {}, {}
    for make in make_decoders(cfg):
        name = make().name
        clean, clean_vx, clean_sm = [], [], []
        sweep = {d: {pkey(p): [] for p in cfg['SER_LEVELS']} for d in CORRUPTORS}
        for k, te in enumerate(folds):
            tr = np.setdiff1d(np.arange(len(trials)), te)
            tr_trials = [trials[i] for i in tr]; te_trials = [trials[i] for i in te]
            Xtr, Ytr, ltr = assemble(tr_trials, L)
            dec = make().fit(Xtr, Ytr, ltr, rng)
            del Xtr
            Xte, Yte, lte = assemble(te_trials, L)
            P = dec.predict(Xte)
            rc = pearson_cols(Yte, P)
            clean.append(rc.mean()); clean_vx.append(rc[0])
            clean_sm.append(pearson_cols(Yte, smooth_per_trial(P, lte, cfg['SMOOTH_SIGMA_SENS'])).mean())
            for d, fn in CORRUPTORS.items():
                for p in cfg['SER_LEVELS']:
                    reps = 1 if p == 0 else cfg['N_CORRUPT_REPEATS']
                    rs = []
                    for _ in range(reps):
                        Xc, Yc, _l = assemble(te_trials, L, fn, p, rng)
                        rs.append(pearson_cols(Yc, dec.predict(Xc)).mean())
                    sweep[d][pkey(p)].append(rs)
            print(f'  {name} fold {k + 1}: r={clean[-1]:.4f}  ({time.time() - t0:.0f}s)')
        R['e1'][name] = dict(r_folds=clean, r_vx_folds=clean_vx, r_smoothed_folds=clean_sm,
                             r=boot_ci(clean, rng), r_vx=boot_ci(clean_vx, rng), r_smoothed=boot_ci(clean_sm, rng),
                             alpha=getattr(dec, 'alpha', None))
        R['e2'][name] = {}
        for d in CORRUPTORS:
            entry, pv = {}, []
            for p in cfg['SER_LEVELS']:
                per_fold = np.array([np.mean(x) for x in sweep[d][pkey(p)]])
                rho_vals = [r / c for rs, c in zip(sweep[d][pkey(p)], clean) for r in rs]
                entry[pkey(p)] = dict(rho=boot_ci(rho_vals, rng), r=float(per_fold.mean()),
                                      p_drop=fisher_paired_p(np.array(clean), per_fold) if p > 0 else 1.0)
                if p > 0:
                    pv.append(entry[pkey(p)]['p_drop'])
            adj, sig = bh(pv)
            for (p, a, s) in zip([q for q in cfg['SER_LEVELS'] if q > 0], adj, sig):
                entry[pkey(p)]['p_bh'] = float(a); entry[pkey(p)]['sig_bh'] = bool(s)
            R['e2'][name][d] = entry

    # E3 — temporal integration (Ridge, per-1ms errors): bin width and smoothing
    R['e3'] = {}
    for rebin in [1, 2, 5]:
        Lr = max(1, (L * cfg['BIN_MS']) // (cfg['BIN_MS'] * rebin))  # keep a 200 ms history
        for sigma in [0, cfg['SMOOTH_SIGMA_SENS']]:
            key = f'bin{cfg["BIN_MS"] * rebin}ms_s{sigma}'
            res = {pkey(p): [] for p in cfg['INTEGRATION_P']}; cl = []
            for te in folds:
                tr = np.setdiff1d(np.arange(len(trials)), te)
                Xtr, Ytr, ltr = assemble([trials[i] for i in tr], Lr, rebin=rebin)
                dec = RidgeDecoder(cfg['RIDGE_ALPHAS']).fit(Xtr, Ytr, ltr, rng)
                te_trials = [trials[i] for i in te]
                Xte, Yte, lte = assemble(te_trials, Lr, rebin=rebin)
                c = pearson_cols(Yte, smooth_per_trial(dec.predict(Xte), lte, sigma)).mean(); cl.append(c)
                for p in cfg['INTEGRATION_P']:
                    for _ in range(cfg['N_CORRUPT_REPEATS']):
                        Xc, Yc, lc = assemble(te_trials, Lr, corrupt_per_1ms, p, rng, rebin=rebin)
                        res[pkey(p)].append(pearson_cols(Yc, smooth_per_trial(dec.predict(Xc), lc, sigma)).mean() / c)
            R['e3'][key] = dict(bin_ms=cfg['BIN_MS'] * rebin, sigma=sigma, lags=Lr, r=boot_ci(cl, rng),
                                rho={k: boot_ci(v, rng) for k, v in res.items()})
            print('E3', key, R['e3'][key]['r'][0])

    # E4 — population size (Ridge, per-1ms errors)
    R['e4'] = {}
    for n in cfg['POP_SIZES']:
        n = min(n, N)
        subsets = [np.arange(N)] if n == N else [np.sort(rng.choice(N, n, replace=False)) for _ in range(cfg['POP_SUBSETS'])]
        res = {pkey(p): [] for p in cfg['INTEGRATION_P']}; cl = []
        for nb in subsets:
            for te in folds:
                tr = np.setdiff1d(np.arange(len(trials)), te)
                Xtr, Ytr, ltr = assemble([trials[i] for i in tr], L, neurons=nb)
                dec = RidgeDecoder(cfg['RIDGE_ALPHAS']).fit(Xtr, Ytr, ltr, rng)
                te_trials = [trials[i] for i in te]
                Xte, Yte, _ = assemble(te_trials, L, neurons=nb)
                c = pearson_cols(Yte, dec.predict(Xte)).mean(); cl.append(c)
                for p in cfg['INTEGRATION_P']:
                    Xc, Yc, _ = assemble(te_trials, L, corrupt_per_1ms, p, rng, neurons=nb)
                    res[pkey(p)].append(pearson_cols(Yc, dec.predict(Xc)).mean() / c)
        R['e4'][str(n)] = dict(r=boot_ci(cl, rng), rho={k: boot_ci(v, rng) for k, v in res.items()})
        print('E4 N=', n, R['e4'][str(n)]['r'][0])

    # E5 — population redundancy: participation ratio of smoothed rates
    def pr(M):
        lam = np.clip(np.linalg.eigvalsh(np.cov(M, rowvar=False)), 0, None)
        return float(lam.sum() ** 2 / (lam ** 2).sum())
    sm = np.vstack([gaussian_filter1d(S.astype(float), 5, axis=0) for S, _ in trials])
    shuf = np.vstack([gaussian_filter1d(S[rng.permutation(len(S))].astype(float), 5, axis=0) for S, _ in trials])
    R['e5'] = dict(pr=pr(sm), pr_time_shuffled=pr(shuf), n_neurons=N)
    print('E5', R['e5'])

    R['runtime_s'] = time.time() - t0
    return R

# %%
# ── Figures, macros, export ─────────────────────────────────────────────
COLORS = {'Ridge': '#2f6db5', 'MLP': '#d08c32', 'LSTM': '#3c9650', 'Transformer': '#b43c3c'}

def make_figures(R, out):
    ps = [p for p in R['config']['SER_LEVELS'] if p > 0]
    # fig0 baseline
    fig, ax = plt.subplots(figsize=(5, 3.2))
    names = list(R['e1'])
    for i, n in enumerate(names):
        f = R['e1'][n]['r_folds']
        ax.bar(i, np.mean(f), color=COLORS.get(n, 'gray'), alpha=.8)
        ax.scatter(np.full(len(f), i), f, color='k', s=10, zorder=3)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names)
    ax.set_ylabel('Pearson r (mean of vx, vy)'); ax.set_ylim(0.5, 1); ax.set_title('Clean decoding, trial-level 5-fold CV')
    fig.tight_layout(); fig.savefig(f'{out}/fig0_baseline.png', dpi=200); plt.close(fig)
    # fig1 SER definitions
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    for d, ls in [('per_spike', '--'), ('per_1ms', '-')]:
        for n in names:
            m = [R['e2'][n][d][pkey(p)]['rho'][0] for p in ps]
            lo = [R['e2'][n][d][pkey(p)]['rho'][1] for p in ps]
            hi = [R['e2'][n][d][pkey(p)]['rho'][2] for p in ps]
            axs[0].plot(ps, m, linestyle=ls, color=COLORS.get(n, 'gray'), marker='o', ms=3)
            axs[0].fill_between(ps, lo, hi, color=COLORS.get(n, 'gray'), alpha=.12)
    axs[0].scatter([1e-3, 1e-3], [PAPER_REF['retention_at_1e-3_M1'], PAPER_REF['retention_at_1e-3_S1']],
                   marker='x', color='k', zorder=4)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=COLORS.get(n, 'gray'), lw=2, label=n) for n in names]
    h += [Line2D([], [], color='0.3', ls='--', label='SER per spike (v2 report)'),
          Line2D([], [], color='0.3', ls='-', label='SER per 1 ms bin'),
          Line2D([], [], color='k', marker='x', ls='', label='Lee et al. 2024 SNN, M1/S1 (different setup)')]
    axs[0].legend(handles=h, fontsize=6.5, loc='lower left')
    axs[0].set_xscale('log'); axs[0].set_xlabel('SER'); axs[0].set_ylabel('retention ρ = r_SER / r_clean')
    axs[0].set_title('Retention depends on how SER is defined')
    fh = R['added_false_hz']
    axs[1].plot(ps, [fh[pkey(p)]['per_spike'] for p in ps], '--o', ms=3, color='0.3', label='per spike')
    axs[1].plot(ps, [fh[pkey(p)]['per_1ms'] for p in ps], '-o', ms=3, color='0', label='per 1 ms bin')
    axs[1].axhline(R['data']['mean_rate_hz'], color='r', lw=.8, ls=':', label='mean true firing rate')
    axs[1].set_xscale('log'); axs[1].set_yscale('log'); axs[1].set_xlabel('SER')
    axs[1].set_ylabel('false spikes added (Hz per neuron)'); axs[1].set_title('Same SER, very different corruption'); axs[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f'{out}/fig1_ser_definitions.png', dpi=200); plt.close(fig)
    # fig2 integration
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    keys = list(R['e3']); x = np.arange(len(keys)); w = 0.38
    for j, p in enumerate(R['config']['INTEGRATION_P']):
        m = [R['e3'][k]['rho'][pkey(p)][0] for k in keys]
        e = [[m[i] - R['e3'][k]['rho'][pkey(p)][1] for i, k in enumerate(keys)],
             [R['e3'][k]['rho'][pkey(p)][2] - m[i] for i, k in enumerate(keys)]]
        ax.bar(x + (j - .5) * w, m, w, yerr=e, label=f'SER {pkey(p)} (per 1 ms)', alpha=.85)
    ax.set_xticks(x); ax.set_xticklabels([f"{R['e3'][k]['bin_ms']} ms\nσ={R['e3'][k]['sigma']}" for k in keys], fontsize=7)
    ax.set_ylabel('retention ρ (Ridge)'); ax.set_title('Temporal integration'); ax.legend(fontsize=7)
    ax.set_ylim(max(0, min(ax.get_ylim()[0], 0.5)), 1.02)
    fig.tight_layout(); fig.savefig(f'{out}/fig2_integration.png', dpi=200); plt.close(fig)
    # fig3 population
    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    ns = [int(n) for n in R['e4']]
    for p in R['config']['INTEGRATION_P']:
        m = [R['e4'][str(n)]['rho'][pkey(p)][0] for n in ns]
        lo = [R['e4'][str(n)]['rho'][pkey(p)][1] for n in ns]; hi = [R['e4'][str(n)]['rho'][pkey(p)][2] for n in ns]
        ax.errorbar(ns, m, yerr=[np.subtract(m, lo), np.subtract(hi, m)], marker='o', capsize=3, label=f'SER {pkey(p)} (per 1 ms)')
    ax.set_xlabel('number of neurons N'); ax.set_ylabel('retention ρ (Ridge)'); ax.set_title('Population size'); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f'{out}/fig3_population.png', dpi=200); plt.close(fig)

def fmt(x, nd=3):
    return 'n/a' if x is None or (isinstance(x, float) and math.isnan(x)) else f'{x:.{nd}f}'

def write_macros(R, out):
    m = {}
    d = R['data']
    m.update({'n_trials': str(d['n_trials']), 'n_neurons': str(d['n_neurons']), 'n_m1': str(d['n_m1']),
              'n_pmd': str(d['n_pmd']), 'sparsity': f"{100 * d['sparsity']:.1f}", 'rate_hz': f"{d['mean_rate_hz']:.2f}",
              'kbar': f"{d['kbar']:.1f}"})
    for p, v in R['added_false_hz'].items():
        m[f'fhz_spike_{p}'] = f"{v['per_spike']:.3g}"; m[f'fhz_1ms_{p}'] = f"{v['per_1ms']:.3g}"
    for n, e in R['e1'].items():
        m[f'r_{n}'] = fmt(e['r'][0]); m[f'r_{n}_ci'] = f"[{fmt(e['r'][1])}, {fmt(e['r'][2])}]"
        m[f'rvx_{n}'] = fmt(e['r_vx'][0]); m[f'rsm_{n}'] = fmt(e['r_smoothed'][0])
    for n, dd in R['e2'].items():
        for dname, entry in dd.items():
            for p, v in entry.items():
                m[f'rho_{n}_{dname}_{p}'] = fmt(v['rho'][0])
                m[f'rho_{n}_{dname}_{p}_ci'] = f"[{fmt(v['rho'][1])}, {fmt(v['rho'][2])}]"
                if 'sig_bh' in v:
                    m[f'sig_{n}_{dname}_{p}'] = 'yes' if v['sig_bh'] else 'no'
    for k, v in R['e3'].items():
        m[f'r_{k}'] = fmt(v['r'][0])
        for p, c in v['rho'].items():
            m[f'rho_{k}_{p}'] = fmt(c[0]); m[f'rho_{k}_{p}_ci'] = f"[{fmt(c[1])}, {fmt(c[2])}]"
    for n, v in R['e4'].items():
        m[f'r_N{n}'] = fmt(v['r'][0])
        for p, c in v['rho'].items():
            m[f'rho_N{n}_{p}'] = fmt(c[0])
    m['pr'] = f"{R['e5']['pr']:.1f}"; m['pr_shuf'] = f"{R['e5']['pr_time_shuffled']:.1f}"
    with open(f'{out}/macros.tex', 'w') as f:
        f.write('% auto-generated by ser_tolerance_v3 — do not edit by hand\n')
        for k, v in m.items():
            f.write('\\expandafter\\def\\csname res@%s\\endcsname{%s}\n' % (k, v))
    return m

def export(R, cfg):
    out = cfg['OUT_DIR']
    make_figures(R, out)
    write_macros(R, out)
    with open(f'{out}/results.json', 'w') as f:
        json.dump(R, f, indent=1, default=float)
    with zipfile.ZipFile('results.zip', 'w') as z:
        for fn in os.listdir(out):
            z.write(os.path.join(out, fn), os.path.join('results', fn))
    print('saved results.zip', os.listdir(out))

# %%
# ── Run ─────────────────────────────────────────────────────────────────
if __name__ == '__main__' or IN_COLAB:
    if os.path.exists(CONFIG['MAT_PATH']):
        RESULTS = run_all(CONFIG)
        export(RESULTS, CONFIG)
        if IN_COLAB:
            from google.colab import files
            files.download('results.zip')
    else:
        print('MAT_PATH not found — edit CONFIG["MAT_PATH"] first:', CONFIG['MAT_PATH'])
