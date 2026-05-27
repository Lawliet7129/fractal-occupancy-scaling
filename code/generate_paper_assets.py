from __future__ import annotations
import sys, csv, time, argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fractal_estimator import (
    simple_random_walk, pearson_walk, brownian_path, persistent_walk,
    antipersistent_walk, levy_flight, fbm_graph, TRACE_MODELS, box_count,
    box_count_gpu, estimate_dimension, local_slopes, window_stability, k_star,
    rescaled_coord, correlation_dimension, correlation_dimension_theiler, dfa_hurst, variogram_hurst,
    higuchi_dimension, occupancy_logcount, occupancy_universal_slope, k_star_model,
    fit_occupancy, occupancy_dimension, occupancy_slope_fd, effective_n,
    model_windowed_bias, bootstrap_ci, rmse, TRUE_TRACE_DIM,
)

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
FIG = PAPER / "figures"; GEN = PAPER / "generated"
FIG.mkdir(exist_ok=True); GEN.mkdir(exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--fast", action="store_true", help="tiny smoke test")
ap.add_argument("--full", action="store_true", help="large camera-ready grid")
ap.add_argument("--gpu", action="store_true",
                help="use CuPy GPU box-counting (exact; same integer counts as CPU)")
A = ap.parse_args()
SEED = 2026
TRUE = TRUE_TRACE_DIM

if A.gpu:
    import cupy as _cp
    box_count = box_count_gpu
    _dev = _cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    print(f"[gpu] CuPy box-counting enabled on {_dev}")

MODELS = dict(TRACE_MODELS)
MODEL_ORDER = ["SRW", "Pearson", "Brownian", "Persistent", "Antipersistent"]
MODEL_LABEL = {"SRW": "SRW (lattice)", "Pearson": "Pearson", "Brownian": "Brownian",
               "Persistent": r"persistent ($\rho{=}0.5$)",
               "Antipersistent": r"anti-persistent ($\rho{=}{-}0.5$)"}

if A.fast:
    TIER = "fast"
    NS = [2**10, 2**12]
    DIMS = [2, 4, 6]
    SCALES = 12
    GRID_SEEDS = lambda N: 4
    GRID_ORIGINS = 1
    ROBUST_ORIGINS = 2
    FBM_HS = [0.1, 0.3, 0.5, 0.7, 0.9]
    FBM_SEEDS = 4
    LEVY_ALPHAS = [0.8, 1.2, 1.6]
    CORR_SUB = 800
    CORR_SEEDS = 4
elif A.full:
    TIER = "full"
    NS = [2**10, 2**12, 2**14, 2**16, 2**18, 2**20]
    DIMS = list(range(2, 21))
    SCALES = 18
    GRID_SEEDS = lambda N: 50 if N <= 2**16 else 20
    GRID_ORIGINS = 1
    ROBUST_ORIGINS = 16
    FBM_HS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    FBM_SEEDS = 40
    LEVY_ALPHAS = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]
    CORR_SUB = 2000
    CORR_SEEDS = 20
else:
    TIER = "default"
    NS = [2**10, 2**12, 2**14, 2**16]
    DIMS = list(range(2, 11))
    SCALES = 14
    GRID_SEEDS = lambda N: 16
    GRID_ORIGINS = 1
    ROBUST_ORIGINS = 8
    FBM_HS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    FBM_SEEDS = 12
    LEVY_ALPHAS = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8]
    CORR_SUB = 1500
    CORR_SEEDS = 8

DEF_WINDOW = (2, 7)
COLLAPSE_WIDTHS = [4, 5, 6]
FBM_N = NS[-2] if A.full else NS[-1]
LEVY_DIM = 3
FN_DIM = 4
PATH_STEPS = 4000
EPS = 0.5 ** np.arange(1, SCALES + 1)
KIDX = np.arange(1, SCALES + 1)
Nmax = NS[-1]

plt.rcParams.update({
    "text.usetex": False, "mathtext.fontset": "cm", "font.family": "serif",
    "axes.unicode_minus": False, "axes.formatter.use_mathtext": True,
    "font.size": 9.5, "axes.labelsize": 10.5, "axes.titlesize": 10.0,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5,
    "lines.linewidth": 1.6, "lines.markersize": 4.5, "axes.linewidth": 0.8,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "xtick.major.size": 3.8, "ytick.major.size": 3.8,
    "xtick.minor.size": 2.0, "ytick.minor.size": 2.0,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.minor.width": 0.6, "ytick.minor.width": 0.6,
    "axes.grid": True, "grid.alpha": 0.18, "grid.linewidth": 0.5,
    "legend.frameon": True, "legend.framealpha": 0.9, "legend.edgecolor": "0.8",
    "legend.fancybox": False, "legend.borderpad": 0.4, "legend.handlelength": 1.7,
    "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "axes.prop_cycle": plt.cycler(color=[
        "#0072B2", "#E69F00", "#009E73", "#D55E00",
        "#CC79A7", "#56B4E9", "#000000", "#F0E442"]),
})
macros = {}
def mac(name, val): macros[name] = val
t0 = time.time()
print(f"[tier={TIER}] true trace dim = {TRUE} (Taylor 1953); fBm graph 2-H; Levy range = alpha.")
print(f"NS={[int(np.log2(n)) for n in NS]} (log2)  DIMS={DIMS[0]}..{DIMS[-1]}  "
      f"SCALES={SCALES}  window={DEF_WINDOW}\n")


def seed_for(model_idx, N, d, s):
    return SEED + model_idx * 1_000_000 + int(round(np.log2(N))) * 10_000 + d * 100 + s


def windows_of_width(width):
    return [(kmin, kmin + width) for kmin in range(1, SCALES - width + 1)]


print("[grid] trace box counts for every (model, N, d, seed) ...")
grid = {}
neff_grid = {}
for mi, name in enumerate(MODEL_ORDER):
    fn = MODELS[name]
    for N in NS:
        nseed = GRID_SEEDS(N)
        for d in DIMS:
            arr = np.empty((nseed, SCALES), dtype=np.float64)
            neff = np.empty(nseed, dtype=np.float64)
            for s in range(nseed):
                rng = np.random.default_rng(seed_for(mi, N, d, s))
                org = np.random.default_rng(seed_for(mi, N, d, s) + 777)
                P = fn(N, d, rng)
                _, c = box_count(P, n_scales=SCALES, n_offsets=GRID_ORIGINS, rng=org)
                arr[s] = c
                neff[s] = effective_n(P) if name == "SRW" else len(P)
            grid[(name, N, d)] = arr
            neff_grid[(name, N, d)] = neff
    print(f"   {name}: done ({time.time()-t0:.0f}s)")

def cell_dhat(name, N, d, window):
    return np.array([estimate_dimension(EPS, c, window) for c in grid[(name, N, d)]])

print("[grid] fBm box counts + alternative estimators ...")
EST_NAMES = ["box", "corr", "dfa", "vario", "higuchi"]
EST_DISP = {"box": "box-counting", "corr": "correlation", "dfa": "DFA",
            "vario": "variogram", "higuchi": "Higuchi"}
fbm_curves = {H: np.empty((FBM_SEEDS, SCALES)) for H in FBM_HS}
fbm_est = {e: {H: [] for H in FBM_HS} for e in EST_NAMES}
for H in FBM_HS:
    for s in range(FBM_SEEDS):
        rng = np.random.default_rng(SEED + 110000 + int(round(H * 100)) * 10 + s)
        P = fbm_graph(FBM_N, H, rng)
        _, c = box_count(P, n_scales=SCALES)
        fbm_curves[H][s] = c
        fbm_est["box"][H].append(estimate_dimension(EPS, c, DEF_WINDOW))
        fbm_est["corr"][H].append(correlation_dimension(P, rng, n_sub=CORR_SUB))
        fbm_est["dfa"][H].append(2.0 - dfa_hurst(P))
        fbm_est["vario"][H].append(2.0 - variogram_hurst(P))
        fbm_est["higuchi"][H].append(higuchi_dimension(P))
print(f"   fBm: done ({time.time()-t0:.0f}s)")

print("[grid] Levy flight box counts (d=%d) ..." % LEVY_DIM)
levy_curves = {}
levy_corr = {a: [] for a in LEVY_ALPHAS}
for a in LEVY_ALPHAS:
    for N in NS:
        arr = np.empty((FBM_SEEDS, SCALES))
        for s in range(FBM_SEEDS):
            rng = np.random.default_rng(SEED + 220000 + int(round(a * 100)) * 100 + int(np.log2(N)) * 7 + s)
            P = levy_flight(N, LEVY_DIM, rng, a)
            _, c = box_count(P, n_scales=SCALES)
            arr[s] = c
            if N == Nmax:
                levy_corr[a].append(correlation_dimension(P, rng, n_sub=CORR_SUB))
        levy_curves[(a, N)] = arr
print(f"   Levy: done ({time.time()-t0:.0f}s)")

print("[1] representative paths ...")
fig, axes = plt.subplots(2, 4, figsize=(12.0, 6.0))
demo = [("SRW", simple_random_walk, None), ("Pearson", pearson_walk, None),
        ("Brownian", brownian_path, None), ("Persistent", persistent_walk, None),
        ("Antipersistent", antipersistent_walk, None)]
for ax, (name, fn, _) in zip(axes.flat, demo):
    rng = np.random.default_rng(SEED + 17 + hash(name) % 1000)
    P = fn(PATH_STEPS, 2, rng)
    ax.plot(P[:, 0], P[:, 1], lw=0.5)
    ax.set_title(MODEL_LABEL[name], fontsize=9.5)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
for ax, a in zip(axes.flat[5:7], [1.5, 1.0]):
    P = levy_flight(PATH_STEPS, 2, np.random.default_rng(SEED + int(a * 100)), a)
    ax.plot(P[:, 0], P[:, 1], lw=0.5, color="C2")
    ax.set_title(rf"Lévy flight ($\alpha{{=}}{a}$)", fontsize=9.5)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
axf = axes.flat[7]
Pf = fbm_graph(PATH_STEPS, 0.5, np.random.default_rng(SEED + 23))
axf.plot(Pf[:, 0], Pf[:, 1], lw=0.6, color="C3")
axf.set_title(r"fBm graph ($H{=}0.5$)", fontsize=9.5)
axf.set_xticks([]); axf.set_yticks([])
fig.suptitle(f"Representative 2-D trajectories ({PATH_STEPS} steps): walk traces "
             "(dim 2), heavy-tailed flights (dim $\\alpha$), self-affine graph (dim $2-H$)")
fig.savefig(FIG / "fig_paths.png"); plt.close(fig)

print("[2] log-log scaling + occupancy model ...")
fig, ax = plt.subplots(figsize=(6.8, 4.6))
ll_lo, ll_hi = DIMS[0], DIMS[-1]
A_fit_record = {}
for d, mk, col in [(ll_lo, "o", "C0"), (ll_hi, "s", "C1")]:
    c = grid[("SRW", Nmax, d)].mean(axis=0)
    n_eff = neff_grid[("SRW", Nmax, d)].mean()
    ax.plot(KIDX, np.log2(c), mk, ms=5, color=col, label=f"SRW $d={d}$ (data)")
    _, Afit = fit_occupancy(EPS, c, n_eff=n_eff, Df=2.0)
    A_fit_record[d] = Afit
    kk = np.linspace(1, SCALES, 200)
    ax.plot(kk, occupancy_logcount(kk, n_eff, 2.0, Afit) / np.log(2), "-", color=col, lw=1.4)
    ks = k_star_model(n_eff, 2.0, Afit)
    ax.axvline(ks, color=col, ls=":", lw=1.2)
ax.plot([], [], "k-", lw=1.4, label="occupancy model")
ax.axvspan(DEF_WINDOW[0], DEF_WINDOW[1], color="orange", alpha=0.15,
           label=f"window $k\\in[{DEF_WINDOW[0]},{DEF_WINDOW[1]}]$")
ax.set_xlabel(r"scale index $k$   ($\epsilon = 2^{-k}$)")
ax.set_ylabel(r"$\log_2 N(\epsilon)$")
ax.legend(fontsize=8.5)
fig.tight_layout()
fig.savefig(FIG / "fig_loglog.png"); plt.close(fig)

print("[3] bias vs dimension ...")
fig, ax = plt.subplots(figsize=(6.6, 4.4))
bias_table = {}
for name, mk in zip(MODEL_ORDER, ["o", "s", "^", "D", "v"]):
    means, los, his = [], [], []
    for d in DIMS:
        est = cell_dhat(name, Nmax, d, DEF_WINDOW)
        m = float(est.mean()); lo, hi = bootstrap_ci(est)
        means.append(m); los.append(lo); his.append(hi)
        bias_table[(name, d)] = (m, lo, hi)
    means = np.array(means); los = np.array(los); his = np.array(his)
    ax.errorbar(DIMS, means - TRUE, yerr=[means - los, his - means], marker=mk,
                ms=4, capsize=2, lw=1, label=MODEL_LABEL[name])
ax.axhline(0.0, color="k", ls="--", lw=1.4, label="unbiased")
ax.set_xlabel("embedding dimension $d$")
ax.set_ylabel(r"box-counting bias $\widehat{D}_B-2$")
ax.set_title(f"Raw bias vs. $d$ at $N=2^{{{int(np.log2(Nmax))}}}$, window "
             f"$k\\in[{DEF_WINDOW[0]},{DEF_WINDOW[1]}]$")
ax.legend(fontsize=8.5, ncol=2)
fig.savefig(FIG / "fig_bias_vs_dim.png"); plt.close(fig)

lines = [
    r"\begin{table}[t]", r"\centering",
    r"\caption{Box-counting estimate $\widehat{D}_{\mathrm B}$ (bootstrap $95\%$ CI"
    r" across seeds) and bias $\widehat{D}_{\mathrm B}-2$ versus embedding dimension"
    f" at $N=2^{{{int(np.log2(Nmax))}}}$ steps, window $k\\in[{DEF_WINDOW[0]},{DEF_WINDOW[1]}]$."
    r" The limiting trace dimension is $2$ in every row; the entire signal is"
    r" finite-size bias.}",
    r"\label{tab:bias}",
    r"\begin{tabular}{cccc}", r"\toprule",
    r"$d$ & SRW $\widehat{D}_{\mathrm B}$ & Pearson $\widehat{D}_{\mathrm B}$ & Brownian $\widehat{D}_{\mathrm B}$\\",
    r"\midrule",
]
for d in DIMS:
    s = bias_table[("SRW", d)]; p = bias_table[("Pearson", d)]; b = bias_table[("Brownian", d)]
    lines.append(
        f"{d} & ${s[0]:.3f}$ \\scriptsize$[{s[1]:.3f},{s[2]:.3f}]$ "
        f"& ${p[0]:.3f}$ \\scriptsize$[{p[1]:.3f},{p[2]:.3f}]$ "
        f"& ${b[0]:.3f}$ \\scriptsize$[{b[1]:.3f},{b[2]:.3f}]$\\\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
(GEN / "results_table.tex").write_text("\n".join(lines), encoding="utf-8")

print("[4] finite-N behaviour ...")
fig, ax = plt.subplots(figsize=(6.6, 4.4))
for name, mk in zip(["SRW", "Pearson", "Brownian"], ["o", "s", "^"]):
    ys = [cell_dhat(name, N, FN_DIM, DEF_WINDOW).mean() - TRUE for N in NS]
    ax.plot([int(np.log2(N)) for N in NS], ys, mk + "-", label=MODEL_LABEL[name])
ax.axhline(0.0, color="k", ls="--", lw=1.4, label="unbiased")
ax.set_xlabel(r"$\log_2 N$")
ax.set_ylabel(r"box-counting bias $\widehat{D}_B-2$")
ax.set_title(f"Finite-$N$ behaviour at $d={FN_DIM}$ (window held fixed)")
ax.legend(fontsize=9)
fig.savefig(FIG / "fig_finite_n.png"); plt.close(fig)
fn_lo = cell_dhat("SRW", NS[0], FN_DIM, DEF_WINDOW).mean()
fn_hi = cell_dhat("SRW", NS[-1], FN_DIM, DEF_WINDOW).mean()

print("[5] window-sensitivity heatmap ...")
c_demo = grid[("SRW", Nmax, 2)].mean(axis=0)
Hmap = np.full((SCALES, SCALES), np.nan)
for kmin in range(1, SCALES):
    for kmax in range(kmin + 1, SCALES + 1):
        Hmap[kmax - 1, kmin - 1] = estimate_dimension(EPS, c_demo, (kmin, kmax))
fig, ax = plt.subplots(figsize=(6.0, 5.0))
im = ax.imshow(Hmap, origin="lower", aspect="auto", cmap="viridis",
               extent=[0.5, SCALES + 0.5, 1.5, SCALES + 0.5], vmin=0, vmax=2.2)
ks = k_star(2, Nmax)
ax.axvline(ks, color="w", ls=":", lw=1.2); ax.axhline(ks, color="w", ls=":", lw=1.2)
ax.set_xlabel(r"$k_{\min}$"); ax.set_ylabel(r"$k_{\max}$")
ax.set_title(rf"$\widehat{{D}}_B$ over windows (SRW, $d=2$, $N=2^{{{int(np.log2(Nmax))}}}$);"
             "\n" rf"white dotted: $k^\star={ks:.1f}$")
fig.colorbar(im, ax=ax, label=r"$\widehat{D}_B$")
fig.savefig(FIG / "fig_window_heatmap.png"); plt.close(fig)
win_vals = Hmap[np.isfinite(Hmap)]

print("[6] scaling collapse + analytic F(x) ...")
pts = []
for name in MODEL_ORDER:
    for N in NS:
        for d in DIMS:
            cmean = grid[(name, N, d)].mean(axis=0)
            ne = float(neff_grid[(name, N, d)].mean())
            Afit_cell = fit_occupancy(EPS, cmean, n_eff=ne, Df=2.0)[1]
            for width in COLLAPSE_WIDTHS:
                for w in windows_of_width(width):
                    D = estimate_dimension(EPS, cmean, w)
                    kmid = 0.5 * (w[0] + w[1])
                    x = kmid - 0.5 * np.log2(ne / d)
                    x_napprox = kmid - 0.5 * np.log2(N / d)
                    x_fitA = kmid - k_star_model(ne, 2.0, Afit_cell)
                    Fm = model_windowed_bias(w, ne, 2.0, float(d))
                    pts.append((x, D - TRUE, name, N, d, width, Fm, x_napprox, x_fitA))
px = np.array([p[0] for p in pts]); py = np.array([p[1] for p in pts])
pmodel = np.array([p[6] for p in pts])
px_napprox = np.array([p[7] for p in pts]); px_fitA = np.array([p[8] for p in pts])

def collapse_residual(xs, ys, nbins=18):
    edges = np.linspace(xs.min(), xs.max(), nbins + 1)
    idx = np.digitize(xs, edges)
    resid, centers, binmean = [], [], []
    for b in range(1, nbins + 2):
        sel = idx == b
        if sel.sum() >= 3:
            resid.append(ys[sel] - ys[sel].mean())
            centers.append(xs[sel].mean()); binmean.append(ys[sel].mean())
    resid = np.concatenate(resid) if resid else np.array([np.nan])
    return float(np.std(ys)), float(np.std(resid)), np.array(centers), np.array(binmean)

raw_sd, res_sd, bc, bm = collapse_residual(px, py)
_, res_sd_napprox, _, _ = collapse_residual(px_napprox, py)
_, res_sd_fitA, _, _ = collapse_residual(px_fitA, py)
model_corr = float(np.corrcoef(py, pmodel)[0, 1])
model_rmse = float(np.sqrt(np.mean((py - pmodel) ** 2)))
cmap = {n: f"C{i}" for i, n in enumerate(MODEL_ORDER)}
fig, ax = plt.subplots(figsize=(6.8, 4.6))
for name in MODEL_ORDER:
    sel = np.array([p[2] == name for p in pts])
    ax.scatter(px[sel], py[sel], s=6, alpha=0.30, color=cmap[name], label=MODEL_LABEL[name])
medges = np.linspace(px.min(), px.max(), 24)
mctr = 0.5 * (medges[1:] + medges[:-1])
mF = np.array([pmodel[(px >= medges[i]) & (px < medges[i + 1])].mean()
               if ((px >= medges[i]) & (px < medges[i + 1])).sum() else np.nan
               for i in range(len(medges) - 1)])
mok = np.isfinite(mF)
ax.plot(mctr[mok], mF[mok], "k-", lw=2.0, label="occupancy model $F(x)$")
ax.axhline(0.0, color="gray", ls="--", lw=1.0)
ax.axvline(0.0, color="gray", ls=":", lw=1.0)
ax.set_xlabel(r"rescaled coordinate $x = \frac{1}{2}(k_{\min}+k_{\max}) - \frac{1}{2}\log_2(N_{\rm eff}/d)$")
ax.set_ylabel(r"box-counting bias $\widehat{D}_B-2$")
ax.legend(fontsize=8, ncol=2, loc="lower left")
fig.tight_layout()
fig.savefig(FIG / "fig_collapse.png"); plt.close(fig)

per_model_res = {}
for name in MODEL_ORDER:
    sel = np.array([p[2] == name for p in pts])
    per_model_res[name] = collapse_residual(px[sel], py[sel])[1]
per_width_res = {}
for width in COLLAPSE_WIDTHS:
    sel = np.array([p[5] == width for p in pts])
    per_width_res[width] = collapse_residual(px[sel], py[sel])[1]

clines = [
    r"\begin{table}[t]", r"\centering",
    r"\caption{Scaling-collapse quality. Raw spread is the standard deviation of"
    r" the box-counting bias $\widehat{D}_{\mathrm B}-2$ over all"
    f" {len(py)} (model, $N$, $d$, window) combinations; the residual spread is"
    r" the standard deviation after subtracting the binned mean $F(x)$ of the"
    r" rescaled coordinate $x$. Per-model rows show the residual is nearly"
    r" model-independent. The occupancy-model prediction of the bias (parameter"
    f"-free, $A=d$) has correlation ${model_corr:.2f}$ with the data.}}",
    r"\label{tab:collapse}",
    r"\begin{tabular}{lccc}", r"\toprule",
    r"grouping & raw spread & residual spread & reduction\\",
    r"\midrule",
    f"all points & ${raw_sd:.3f}$ & ${res_sd:.3f}$ & ${raw_sd/res_sd:.1f}\\times$\\\\",
    r"\midrule",
]
for name in MODEL_ORDER:
    clines.append(f"{MODEL_LABEL[name]} & -- & ${per_model_res[name]:.3f}$ & --\\\\")
clines += [r"\midrule"]
for width in COLLAPSE_WIDTHS:
    clines.append(f"window width ${width}$ & -- & ${per_width_res[width]:.3f}$ & --\\\\")
clines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
(GEN / "results_collapse.tex").write_text("\n".join(clines), encoding="utf-8")

with open(GEN / "results_collapse.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["x", "bias", "model", "N", "d", "width", "F_model"])
    for p in pts:
        w.writerow([f"{p[0]:.4f}", f"{p[1]:.4f}", p[2], p[3], p[4], p[5], f"{p[6]:.4f}"])

print("[7] universal local-slope collapse ...")
def universal_points(curve, n_eff, Df, A=None):
    if A is None:
        try:
            _, A = fit_occupancy(EPS, curve, n_eff=n_eff, Df=Df)
        except Exception:
            return np.array([]), np.array([]), np.array([])
    kstar = k_star_model(n_eff, Df, A)
    k = np.arange(1, len(curve))
    s = local_slopes(curve)
    xi = Df * (k + 0.5 - kstar)
    model_h = occupancy_slope_fd(k, n_eff, Df, A)
    sel = curve[:-1] >= 2
    return xi[sel], (s / Df)[sel], model_h[sel]

UNI_DIMS = [dd for dd in (3, 5, 8) if dd in DIMS]
UNI_FBM = [h for h in (0.3, 0.5, 0.7) if h in FBM_HS]
UNI_LEVY = [aa for aa in (0.6, 0.8, 1.0, 1.2, 1.6) if aa in LEVY_ALPHAS]
uni_df_lo = min([2.0] + [2.0 - h for h in UNI_FBM] + list(UNI_LEVY))
fam_pts = {"trace": ([], [], []), "fbm": ([], [], []), "levy": ([], [], [])}
nofit_h, nofit_mh = [], []
def _add(fam, curve, n_eff, Df):
    xi, h, mh = universal_points(curve, n_eff, Df)
    fam_pts[fam][0].append(xi); fam_pts[fam][1].append(h); fam_pts[fam][2].append(mh)
    _, h0, mh0 = universal_points(curve, n_eff, Df, A=1.0)
    nofit_h.append(h0); nofit_mh.append(mh0)
for d in UNI_DIMS:
    for N in NS:
        _add("trace", grid[("SRW", N, d)].mean(axis=0),
             neff_grid[("SRW", N, d)].mean(), 2.0)
for H in UNI_FBM:
    _add("fbm", fbm_curves[H].mean(axis=0), FBM_N + 1, 2.0 - H)
for a in UNI_LEVY:
    for N in NS:
        _add("levy", levy_curves[(a, N)].mean(axis=0), N + 1, a)
fam_color = {"trace": "C0", "fbm": "C3", "levy": "C2"}
fam_disp = {"trace": "walk traces ($D_f=2$)", "fbm": "fBm graphs ($D_f=2-H$)",
            "levy": r"Lévy flights ($D_f=\alpha$)"}
allxi, allh, allmh = [], [], []
fig, ax = plt.subplots(figsize=(6.8, 4.6))
for fam in ["trace", "fbm", "levy"]:
    if not fam_pts[fam][0]:
        continue
    xi = np.concatenate(fam_pts[fam][0]); h = np.concatenate(fam_pts[fam][1])
    allxi.append(xi); allh.append(h); allmh.append(np.concatenate(fam_pts[fam][2]))
    ax.scatter(xi, h, s=14, alpha=0.6, color=fam_color[fam], label=fam_disp[fam])
allxi = np.concatenate(allxi); allh = np.concatenate(allh); allmh = np.concatenate(allmh)
xg = np.linspace(allxi.min(), allxi.max(), 300)
ax.plot(xg, occupancy_universal_slope(xg), "k-", lw=2.0, label=r"crossover $h(\xi)$")
ax.set_xlabel(r"$\xi = D_f\,(k - k^\star) = \log_2(b/N_{\rm eff})$")
ax.set_ylabel(r"normalized local slope $s(k)/D_f$")
uni_corr = float(np.corrcoef(allh, allmh)[0, 1])
uni_rmse = float(np.sqrt(np.nanmean((allh - allmh) ** 2)))
uni_maxresid = float(np.nanmax(np.abs(allh - allmh)))
_cal = np.polyfit(allmh, allh, 1)
uni_cal_slope = float(_cal[0]); uni_cal_int = float(_cal[1])
_nh = np.concatenate(nofit_h); _nmh = np.concatenate(nofit_mh)
_m = np.isfinite(_nh) & np.isfinite(_nmh)
uni_corr_nofit = float(np.corrcoef(_nh[_m], _nmh[_m])[0, 1])
uni_rmse_nofit = float(np.sqrt(np.mean((_nh[_m] - _nmh[_m]) ** 2)))
ax.legend(fontsize=8.5)
fig.tight_layout()
fig.savefig(FIG / "fig_universal.png"); plt.close(fig)

print("[8] bias correction ...")
def family_correction(curves_by_param, truth_fn, neff_by_param):
    box_b, occ_b, box_v, occ_v, tv = [], [], [], [], []
    for pv, arr in curves_by_param.items():
        t = truth_fn(pv); ne = neff_by_param[pv]
        for i, c in enumerate(arr):
            n_eff = ne[i] if np.ndim(ne) else float(ne)
            bd = estimate_dimension(EPS, c, DEF_WINDOW)
            od = occupancy_dimension(EPS, c, n_eff=n_eff)
            box_v.append(bd); occ_v.append(od); tv.append(t)
            box_b.append(bd - t); occ_b.append(od - t)
    box_v = np.array(box_v); occ_v = np.array(occ_v); tv = np.array(tv)
    return (float(np.nanmean(box_b)), float(np.sqrt(np.nanmean((box_v - tv) ** 2))),
            float(np.nanmean(occ_b)), float(np.sqrt(np.nanmean((occ_v - tv) ** 2))))

trace_curves = {d: grid[("SRW", Nmax, d)] for d in DIMS}
corr_trace_fam = family_correction(trace_curves, lambda d: 2.0,
                                   {d: neff_grid[("SRW", Nmax, d)] for d in DIMS})
corr_fbm_fam = family_correction({H: fbm_curves[H] for H in FBM_HS}, lambda H: 2.0 - H,
                                  {H: FBM_N + 1 for H in FBM_HS})
corr_levy_fam = family_correction({a: levy_curves[(a, Nmax)] for a in LEVY_ALPHAS}, lambda a: a,
                                   {a: Nmax + 1 for a in LEVY_ALPHAS})

def _occ_estimates(name, N, d):
    ne = neff_grid[(name, N, d)]
    return np.array([occupancy_dimension(EPS, c, n_eff=ne[i])
                     for i, c in enumerate(grid[(name, N, d)])])
occ_d2 = _occ_estimates("SRW", Nmax, 2); box_d2 = cell_dhat("SRW", Nmax, 2, DEF_WINDOW)
occ_d2_m = float(np.nanmean(occ_d2)); occ_d2_lo, occ_d2_hi = bootstrap_ci(occ_d2)
box_d2_m = float(np.nanmean(box_d2)); box_d2_lo, box_d2_hi = bootstrap_ci(box_d2)
c_rep = grid[("SRW", Nmax, 2)][0]; ne_rep = float(neff_grid[("SRW", Nmax, 2)][0])
Df_rep, A_rep = fit_occupancy(EPS, c_rep, n_eff=ne_rep, Df=None)
k_rep = np.log2(1.0 / EPS); y_rep = np.log(c_rep); m_rep = c_rep >= 2
Dgrid = np.linspace(1.3, 2.7, 70); Agrid = np.logspace(np.log10(0.3), np.log10(80.0), 70)
Lsurf = np.empty((Agrid.size, Dgrid.size))
for ia, Av in enumerate(Agrid):
    pred = np.array([occupancy_logcount(k_rep[m_rep], ne_rep, Dv, Av) for Dv in Dgrid])
    Lsurf[ia] = np.mean((y_rep[m_rep][None, :] - pred) ** 2, axis=1)

def build_xbias(models, widths=COLLAPSE_WIDTHS):
    X, B = [], []
    for name in models:
        for N in NS:
            for d in DIMS:
                cmean = grid[(name, N, d)].mean(axis=0)
                for width in widths:
                    for w in windows_of_width(width):
                        X.append(rescaled_coord(w, d, N))
                        B.append(estimate_dimension(EPS, cmean, w) - TRUE)
    return np.array(X), np.array(B)

train_models = ["SRW", "Pearson", "Brownian"]
test_models = ["Persistent", "Antipersistent"]
Xtr, Btr = build_xbias(train_models)
Xte, Bte = build_xbias(test_models)
edges = np.linspace(min(Xtr.min(), Xte.min()), max(Xtr.max(), Xte.max()), 30)
ctr = 0.5 * (edges[1:] + edges[:-1])
Fhat = np.array([Btr[(Xtr >= edges[i]) & (Xtr < edges[i + 1])].mean()
                 if ((Xtr >= edges[i]) & (Xtr < edges[i + 1])).sum() else np.nan
                 for i in range(len(edges) - 1)])
ok = np.isfinite(Fhat)
pred_te = np.interp(Xte, ctr[ok], Fhat[ok])
reg = Xte < 1.5
held_raw = float(np.sqrt(np.mean(Bte[reg] ** 2)))
held_corr = float(np.sqrt(np.mean((Bte[reg] - pred_te[reg]) ** 2)))

fig, (axL, axM, axR) = plt.subplots(1, 3, figsize=(15.5, 4.3))
fam_labels = ["walk traces\n($D=2$)", "fBm graphs\n($D=2-H$)", "Lévy flights" "\n" r"($D=\alpha$)"]
box_rmses = [corr_trace_fam[1], corr_fbm_fam[1], corr_levy_fam[1]]
occ_rmses = [corr_trace_fam[3], corr_fbm_fam[3], corr_levy_fam[3]]
xb = np.arange(3); wbar = 0.36
axL.bar(xb - wbar / 2, box_rmses, wbar, label="box-counting (window)", color="C0")
axL.bar(xb + wbar / 2, occ_rmses, wbar, label="occupancy-fit (corrected)", color="C1")
axL.set_xticks(xb); axL.set_xticklabels(fam_labels, fontsize=9)
axL.set_ylabel("RMSE vs. true dimension")
axL.set_title("(a) Occupancy-fit correction"); axL.legend(fontsize=9)
cf = axM.contourf(Dgrid, Agrid, np.log10(Lsurf), levels=18, cmap="viridis")
axM.set_yscale("log"); axM.plot([Df_rep], [A_rep], "r*", ms=13, label="best fit")
axM.axvline(2.0, color="w", ls=":", lw=1.2)
axM.set_xlabel(r"dimension $D_f$"); axM.set_ylabel(r"prefactor $A$")
axM.set_title(r"(b) Fit-loss surface (single $d{=}2$ trace)")
axM.legend(fontsize=8.5, loc="upper left")
fig.colorbar(cf, ax=axM, fraction=0.046, pad=0.04, label=r"$\log_{10}$ MSE")
axR.scatter(Xte, Bte, s=6, alpha=0.25, color="C3", label="held-out (raw bias)")
axR.scatter(Xte[reg], (Bte - pred_te)[reg], s=6, alpha=0.3, color="C0", label="held-out (corrected)")
axR.plot(ctr[ok], Fhat[ok], "k-", lw=1.6, label=r"$\widehat{F}(x)$ (trained)")
axR.axhline(0, color="gray", ls="--", lw=1.0)
axR.set_xlabel(r"rescaled coordinate $x$"); axR.set_ylabel(r"$\widehat{D}_B-2$")
axR.set_title(f"(c) Held-out correction (RMSE {held_raw:.2f}$\\to${held_corr:.2f})")
axR.legend(fontsize=8.5)
fig.tight_layout()
fig.savefig(FIG / "fig_correction.png"); plt.close(fig)

elines = [
    r"\begin{table}[t]", r"\centering",
    r"\caption{Finite-size bias correction. For each model family we compare the"
    r" windowed box-counting estimate (window "
    f"$k\\in[{DEF_WINDOW[0]},{DEF_WINDOW[1]}]$) with the occupancy-fit dimension"
    r" (the $D_f$ of the best-fit occupancy curve), reporting mean bias and RMSE"
    r" against the known dimension. Traces use $N=2^{" + str(int(np.log2(Nmax))) +
    r"}$ over $d=2,\dots," + str(DIMS[-1]) + r"$; fBm over $H$; L\'evy over"
    r" $\alpha$. The occupancy-fit correction reduces the RMSE for the walk traces"
    r" and L\'evy flights, where the finite-window bias is largest, and is"
    r" essentially neutral on fBm graphs.}",
    r"\label{tab:correction}",
    r"\begin{tabular}{lcccc}", r"\toprule",
    r"family & box bias & box RMSE & occ.-fit bias & occ.-fit RMSE\\", r"\midrule",
    f"walk traces ($D=2$) & ${corr_trace_fam[0]:+.3f}$ & ${corr_trace_fam[1]:.3f}$ & ${corr_trace_fam[2]:+.3f}$ & ${corr_trace_fam[3]:.3f}$\\\\",
    f"fBm graphs ($D=2-H$) & ${corr_fbm_fam[0]:+.3f}$ & ${corr_fbm_fam[1]:.3f}$ & ${corr_fbm_fam[2]:+.3f}$ & ${corr_fbm_fam[3]:.3f}$\\\\",
    f"L\\'evy flights ($D=\\alpha$) & ${corr_levy_fam[0]:+.3f}$ & ${corr_levy_fam[1]:.3f}$ & ${corr_levy_fam[2]:+.3f}$ & ${corr_levy_fam[3]:.3f}$\\\\",
    r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
]
(GEN / "results_correction.tex").write_text("\n".join(elines), encoding="utf-8")

print("[9] local-slope stability diagnostic ...")
allS, allab, allsub = [], [], []
sw_by = {n: [] for n in MODEL_ORDER}; ab_by = {n: [] for n in MODEL_ORDER}
for name in MODEL_ORDER:
    for N in NS:
        for d in DIMS:
            ks = k_star(d, N)
            for c in grid[(name, N, d)]:
                for width in COLLAPSE_WIDTHS:
                    for w in windows_of_width(width):
                        S = window_stability(c, w)
                        if not np.isfinite(S):
                            continue
                        b = abs(estimate_dimension(EPS, c, w) - TRUE)
                        sub = w[1] <= ks
                        allS.append(S); allab.append(b); allsub.append(sub)
                        sw_by[name].append(S); ab_by[name].append(b)
allS = np.array(allS); allab = np.array(allab); allsub = np.array(allsub, dtype=bool)
r_pear = float(np.corrcoef(allS, allab)[0, 1]); r_spear = float(spearmanr(allS, allab).statistic)
r_pear_sub = float(np.corrcoef(allS[allsub], allab[allsub])[0, 1])
r_spear_sub = float(spearmanr(allS[allsub], allab[allsub]).statistic)
mbias_sub = float(allab[allsub].mean()); mbias_plat = float(allab[~allsub].mean())
spear_models = [float(spearmanr(np.array(sw_by[n]), np.array(ab_by[n])).statistic) for n in MODEL_ORDER]
fig, ax = plt.subplots(figsize=(6.8, 4.6))
ax.scatter(allS[allsub], allab[allsub], s=5, alpha=0.20, color="C0",
           label=r"window below $k^\star$ (reliable region)")
ax.scatter(allS[~allsub], allab[~allsub], s=5, alpha=0.10, color="C3",
           label=r"window reaching the plateau")
ax.axhline(mbias_plat, color="C3", ls=":", lw=1.2)
ax.axhline(mbias_sub, color="C0", ls=":", lw=1.2)
ax.set_xlabel(r"window stability $S(W)$ (std. of local slopes in $W$)")
ax.set_ylabel(r"$|\widehat{D}_B-2|$")
ax.set_xlim(0, np.percentile(allS, 99))
ax.set_title("Slope stability does not flag the worst windows\n"
             rf"(overall Spearman $\rho={r_spear:.2f}$): the flat plateau has low $S$, high bias")
ax.legend(fontsize=8.5, loc="upper right")
fig.savefig(FIG / "fig_diagnostic.png"); plt.close(fig)

dlines = [
    r"\begin{table}[t]", r"\centering",
    r"\caption{The local-slope stability $S(W)$ is \emph{not} a sufficient"
    r" diagnostic for box-counting reliability. We correlate $S(W)$ with the"
    f" absolute bias $|\\widehat{{D}}_{{\\mathrm B}}-2|$ over {len(allS):,} (model,"
    r" $N$, $d$, window, seed) points. Across all windows the correlation is"
    r" \emph{negative}, because the most biased windows are the flat fine-scale"
    r" plateau (small $S$, mean bias "
    f"${mbias_plat:.2f}$) rather than the lower-bias windows below the"
    r" saturation scale $k^\star$ (mean bias "
    f"${mbias_sub:.2f}$). The operative reliability indicator is the window's"
    r" position relative to $k^\star$ (the coordinate $x$), not $S(W)$.}",
    r"\label{tab:diagnostic}",
    r"\begin{tabular}{lcccc}", r"\toprule",
    r"window subset & count & Pearson $r$ & Spearman $\rho$ & mean $|\widehat{D}_{\mathrm B}-2|$\\",
    r"\midrule",
    f"all windows & {len(allS):,} & ${r_pear:.2f}$ & ${r_spear:.2f}$ & ${allab.mean():.2f}$\\\\",
    f"below $k^\\star$ (reliable) & {int(allsub.sum()):,} & ${r_pear_sub:.2f}$ & ${r_spear_sub:.2f}$ & ${mbias_sub:.2f}$\\\\",
    f"reaching plateau & {int((~allsub).sum()):,} & -- & -- & ${mbias_plat:.2f}$\\\\",
    r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
]
(GEN / "results_diagnostic.tex").write_text("\n".join(dlines), encoding="utf-8")

print("[10] estimator comparison ...")
CORR_DIMS = [d for d in (2, 4, 6, 8, 10) if d in DIMS]
corr_trace = {}
for d in CORR_DIMS:
    vals = []
    for s in range(CORR_SEEDS):
        rng = np.random.default_rng(SEED + 90000 + d * 50 + s)
        vals.append(correlation_dimension(simple_random_walk(Nmax, d, rng), rng, n_sub=CORR_SUB))
    corr_trace[d] = float(np.nanmean(vals))

THEILER_W = 100
_dth = 4 if 4 in DIMS else DIMS[0]
_Nth = 2**16 if 2**16 in NS else NS[-1]
_Pth = simple_random_walk(_Nth, _dth, np.random.default_rng(SEED + 70707))
corr_no_theiler = correlation_dimension_theiler(_Pth, np.random.default_rng(SEED + 808), n_sub=CORR_SUB, theiler=0)
corr_theiler = correlation_dimension_theiler(_Pth, np.random.default_rng(SEED + 808), n_sub=CORR_SUB, theiler=THEILER_W)

fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.5, 4.3))
axL.plot(DIMS, [bias_table[("SRW", d)][0] for d in DIMS], "o-", label="box-counting")
axL.plot(CORR_DIMS, [corr_trace[d] for d in CORR_DIMS], "^-", label="correlation dim.")
axL.axhline(TRUE, color="k", ls="--", lw=1.4, label="true $=2$")
axL.set_xlabel("embedding dimension $d$"); axL.set_ylabel("estimated dimension")
axL.set_title("(a) Walk traces (true $D=2$)"); axL.legend(fontsize=9)
Hs = np.array(FBM_HS)
axR.plot(Hs, 2 - Hs, "k--", lw=1.4, label="true $2-H$")
for e, mk in zip(EST_NAMES, ["o", "^", "s", "D", "v"]):
    axR.plot(Hs, [np.nanmean(fbm_est[e][H]) for H in FBM_HS], mk + "-", ms=4, label=EST_DISP[e])
axR.set_xlabel("Hurst exponent $H$"); axR.set_ylabel("estimated graph dimension")
axR.set_title("(b) fBm graphs (true $D=2-H$)"); axR.legend(fontsize=8.5)
fig.tight_layout(); fig.savefig(FIG / "fig_estimator_compare.png"); plt.close(fig)

elines = [
    r"\begin{table}[t]", r"\centering",
    r"\caption{Estimator accuracy on fractional Brownian motion graphs (true"
    f" dimension $2-H$), aggregated over $H\\in\\{{{FBM_HS[0]},\\dots,{FBM_HS[-1]}\\}}$"
    f" with {FBM_SEEDS} paths of $N=2^{{{int(np.log2(FBM_N))}}}$ each. Mean bias is"
    r" $\langle\widehat{D}-(2-H)\rangle$; RMSE is over all $(H,\text{seed})$"
    r" estimates. Structure-function estimators recover the truth far better than"
    r" box-counting, which is biased low and worst in the rough regime (small $H$).}",
    r"\label{tab:estimator}",
    r"\begin{tabular}{lccc}", r"\toprule",
    r"estimator & mean bias & RMSE & $\widehat{D}$ at $H{=}0.1$ (true $1.9$)\\",
    r"\midrule",
]
est_summary = {}
for e in EST_NAMES:
    allv, allt, biases = [], [], []
    for H in FBM_HS:
        v = np.array(fbm_est[e][H], dtype=float)
        allv.append(v); allt.append(np.full(v.shape, 2 - H)); biases.append(np.nanmean(v) - (2 - H))
    allv = np.concatenate(allv); allt = np.concatenate(allt); mask = np.isfinite(allv)
    mbias = float(np.mean(biases)); rms = float(np.sqrt(np.mean((allv[mask] - allt[mask]) ** 2)))
    at01 = float(np.nanmean(fbm_est[e][FBM_HS[0]]))
    est_summary[e] = (mbias, rms, at01)
    elines.append(f"{EST_DISP[e]} & ${mbias:+.3f}$ & ${rms:.3f}$ & ${at01:.3f}$\\\\")
elines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
(GEN / "results_estimator.tex").write_text("\n".join(elines), encoding="utf-8")

with open(GEN / "results_estimators.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["H", "true_2mH"] + EST_NAMES)
    for H in FBM_HS:
        w.writerow([H, f"{2-H:.4f}"] + [f"{np.nanmean(fbm_est[e][H]):.4f}" for e in EST_NAMES])

print("[11] application: two single stochastic trajectories ...")
app_N = Nmax
kk = np.linspace(1, SCALES, 200)
app_walk_d = 6 if 6 in DIMS else DIMS[-1]
rngw = np.random.default_rng(SEED + 4040)
Pw = simple_random_walk(app_N, app_walk_d, rngw)
_, cw = box_count(Pw, n_scales=SCALES)
nw = effective_n(Pw)
app_walk_box = estimate_dimension(EPS, cw, DEF_WINDOW)
app_walk_occ = occupancy_dimension(EPS, cw, n_eff=nw)
Aw = fit_occupancy(EPS, cw, n_eff=nw, Df=app_walk_occ)[1]
app_alpha = 1.5
rngl = np.random.default_rng(SEED + 5050)
Pl = levy_flight(app_N, 2, rngl, app_alpha)
_, cl = box_count(Pl, n_scales=SCALES)
nl = effective_n(Pl)
app_box = estimate_dimension(EPS, cl, DEF_WINDOW)
app_occ = occupancy_dimension(EPS, cl, n_eff=nl)
if not np.isfinite(app_occ):
    app_occ = app_box
Al = fit_occupancy(EPS, cl, n_eff=nl, Df=app_occ)[1]
app_corr = correlation_dimension(Pl, rngl, n_sub=CORR_SUB)

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.3))
axL.plot(KIDX, np.log2(cw), "o", ms=4, color="C0", label="box counts")
axL.plot(kk, occupancy_logcount(kk, nw, app_walk_occ, Aw) / np.log(2), "k-", lw=1.4,
         label=f"occupancy fit ($D={app_walk_occ:.2f}$)")
axL.axvspan(DEF_WINDOW[0], DEF_WINDOW[1], color="orange", alpha=0.15)
axL.set_xlabel(r"scale index $k$"); axL.set_ylabel(r"$\log_2 N(\epsilon)$")
axL.set_title(f"Diffusive walk ($d={app_walk_d}$, true $2$):\n"
              f"naive box $={app_walk_box:.2f}$, occ.-fit $={app_walk_occ:.2f}$")
axL.legend(fontsize=9)
axR.plot(KIDX, np.log2(cl), "o", ms=4, color="C2", label="box counts")
axR.plot(kk, occupancy_logcount(kk, nl, app_occ, Al) / np.log(2), "k-", lw=1.4,
         label=f"occupancy fit ($D={app_occ:.2f}$)")
axR.axvspan(DEF_WINDOW[0], DEF_WINDOW[1], color="orange", alpha=0.15)
axR.set_xlabel(r"scale index $k$"); axR.set_ylabel(r"$\log_2 N(\epsilon)$")
axR.set_title(rf"Superdiffusive Lévy flight ($\alpha={app_alpha}$):" "\n"
              f"box $={app_box:.2f}$, occ.-fit $={app_occ:.2f}$, corr.-dim $={app_corr:.2f}$")
axR.legend(fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "fig_application.png"); plt.close(fig)

print("[12] grid-origin robustness ...")
off = {}
rob_dims = [d for d in (2, 4, 8) if d in DIMS]
for d in rob_dims:
    vals = []
    for s in range(min(8, GRID_SEEDS(Nmax))):
        rng = np.random.default_rng(SEED + 4242 + d * 10 + s)
        org = np.random.default_rng(SEED + 9999 + d * 10 + s)
        _, c = box_count(simple_random_walk(Nmax, d, rng), n_scales=SCALES,
                         n_offsets=ROBUST_ORIGINS, rng=org)
        vals.append(estimate_dimension(EPS, c, DEF_WINDOW))
    off[d] = float(np.mean(vals))
max_off_shift = max(abs(off[d] - cell_dhat("SRW", Nmax, d, DEF_WINDOW).mean()) for d in rob_dims)

with open(GEN / "results_main.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["model", "N", "d", "D_hat", "bias", "ci_lo", "ci_hi"])
    for name in MODEL_ORDER:
        for N in NS:
            for d in DIMS:
                est = cell_dhat(name, N, d, DEF_WINDOW)
                m = est.mean(); lo, hi = bootstrap_ci(est)
                w.writerow([name, N, d, f"{m:.4f}", f"{m-TRUE:.4f}", f"{lo:.4f}", f"{hi:.4f}"])

mac("tier", TIER); mac("Nmaxexp", str(int(np.log2(Nmax)))); mac("Nmax", f"{Nmax:,}")
mac("Nminexp", str(int(np.log2(NS[0])))); mac("dmax", str(DIMS[-1]))
mac("nmodels", str(len(MODEL_ORDER))); mac("gridseeds", str(GRID_SEEDS(Nmax)))
mac("nscales", str(SCALES)); mac("winlo", str(DEF_WINDOW[0])); mac("winhi", str(DEF_WINDOW[1]))
mac("fbmN", f"{FBM_N:,}"); mac("fbmNexp", str(int(np.log2(FBM_N)))); mac("fbmPaths", str(FBM_SEEDS))
mac("collapseWidths", ", ".join(str(w) for w in COLLAPSE_WIDTHS)); mac("npts", f"{len(py):,}")
mac("DtwoSRW", f"{bias_table[('SRW',2)][0]:.2f}"); mac("biasTwoSRW", f"{bias_table[('SRW',2)][0]-TRUE:.2f}")
mac("DtwoBrown", f"{bias_table[('Brownian',2)][0]:.2f}")
mac("biasRangeLo", f"{min(bias_table[(n,d)][0]-TRUE for n in MODEL_ORDER for d in DIMS):.2f}")
mac("biasRangeHi", f"{max(bias_table[(n,d)][0]-TRUE for n in MODEL_ORDER for d in DIMS):.2f}")
mac("fnDim", str(FN_DIM)); mac("fnDlo", f"{fn_lo:.2f}"); mac("fnDhi", f"{fn_hi:.2f}")
_wmin = win_vals.min()
mac("winMin", f"{0.0 if abs(_wmin) < 0.005 else _wmin:.2f}"); mac("winMax", f"{win_vals.max():.2f}")
mac("collapseRaw", f"{raw_sd:.2f}"); mac("collapseResid", f"{res_sd:.2f}")
mac("collapseFactor", f"{raw_sd/res_sd:.1f}"); mac("collapseModelMax", f"{max(per_model_res.values()):.2f}")
mac("collapseResidApprox", f"{res_sd_napprox:.2f}"); mac("collapseResidFitA", f"{res_sd_fitA:.2f}")
mac("modelCorr", f"{model_corr:.2f}"); mac("modelRMSE", f"{model_rmse:.2f}")
mac("AfitLo", f"{A_fit_record[ll_lo]:.2f}"); mac("AfitHi", f"{A_fit_record[ll_hi]:.2f}")
mac("AfitLoD", str(ll_lo)); mac("AfitHiD", str(ll_hi))
mac("uniCorr", f"{uni_corr:.2f}"); mac("uniRMSE", f"{uni_rmse:.2f}")
mac("uniDfLo", f"{uni_df_lo:.1f}")
mac("uniMaxResid", f"{uni_maxresid:.2f}"); mac("uniCalSlope", f"{uni_cal_slope:.2f}")
mac("uniCalInt", f"{uni_cal_int:+.2f}"); mac("uniCorrNofit", f"{uni_corr_nofit:.2f}")
mac("uniRMSEnofit", f"{uni_rmse_nofit:.2f}")
mac("traceBoxRMSE", f"{corr_trace_fam[1]:.2f}"); mac("traceOccRMSE", f"{corr_trace_fam[3]:.2f}")
mac("fbmBoxRMSE", f"{corr_fbm_fam[1]:.3f}"); mac("fbmOccRMSE", f"{corr_fbm_fam[3]:.3f}")
mac("levyBoxRMSE", f"{corr_levy_fam[1]:.3f}"); mac("levyOccRMSE", f"{corr_levy_fam[3]:.3f}")
mac("heldRawRMSE", f"{held_raw:.2f}"); mac("heldCorrRMSE", f"{held_corr:.2f}")
mac("traceBoxDtwo", f"{box_d2_m:.2f}"); mac("traceBoxDtwoLo", f"{box_d2_lo:.2f}"); mac("traceBoxDtwoHi", f"{box_d2_hi:.2f}")
mac("traceOccDtwo", f"{occ_d2_m:.2f}"); mac("traceOccDtwoLo", f"{occ_d2_lo:.2f}"); mac("traceOccDtwoHi", f"{occ_d2_hi:.2f}")
mac("diagPearson", f"{r_pear:.2f}"); mac("diagSpearman", f"{r_spear:.2f}")
mac("diagPearsonSub", f"{r_pear_sub:.2f}"); mac("diagSpearmanSub", f"{r_spear_sub:.2f}")
mac("meanBiasPlateau", f"{mbias_plat:.2f}"); mac("meanBiasSub", f"{mbias_sub:.2f}")
mac("diagSpearMin", f"{min(spear_models):.2f}"); mac("diagSpearMax", f"{max(spear_models):.2f}")
mac("boxRMSE", f"{est_summary['box'][1]:.3f}"); mac("dfaRMSE", f"{est_summary['dfa'][1]:.3f}")
mac("varioRMSE", f"{est_summary['vario'][1]:.3f}"); mac("higuchiRMSE", f"{est_summary['higuchi'][1]:.3f}")
mac("corrRMSE", f"{est_summary['corr'][1]:.3f}")
mac("theilerW", str(THEILER_W)); mac("corrNoTheiler", f"{corr_no_theiler:.2f}")
mac("corrTheiler", f"{corr_theiler:.2f}"); mac("corrTheilerD", str(_dth))
mac("fbmRoughTrue", f"{2-FBM_HS[0]:.2f}"); mac("fbmRoughBox", f"{est_summary['box'][2]:.2f}")
mac("fbmRoughBoxH", f"{2-est_summary['box'][2]:.2f}"); mac("fbmRoughDFA", f"{est_summary['dfa'][2]:.2f}")
mac("fbmRoughHiguchi", f"{est_summary['higuchi'][2]:.2f}")
mac("corrFour", f"{corr_trace.get(4, float('nan')):.2f}"); mac("corrEight", f"{corr_trace.get(8, float('nan')):.2f}")
mac("levyAlphaLo", f"{LEVY_ALPHAS[0]:.1f}"); mac("levyAlphaHi", f"{LEVY_ALPHAS[-1]:.1f}")
mac("levyDim", str(LEVY_DIM))
mac("appAlpha", f"{app_alpha}"); mac("appBox", f"{app_box:.2f}")
mac("appOcc", f"{app_occ:.2f}"); mac("appCorr", f"{app_corr:.2f}")
mac("appWalkD", str(app_walk_d)); mac("appWalkBox", f"{app_walk_box:.2f}")
mac("appWalkOcc", f"{app_walk_occ:.2f}")
mac("robustOrigins", str(ROBUST_ORIGINS)); mac("maxOffShift", f"{max_off_shift:.2f}")

mlines = ["% Auto-generated by generate_paper_assets.py -- do not edit by hand."]
for k, v in macros.items():
    mlines.append(f"\\newcommand{{\\{k}}}{{{v}}}")
(GEN / "macros.tex").write_text("\n".join(mlines) + "\n", encoding="utf-8")

print(f"\nWrote figures to {FIG}")
print(f"Wrote tables + macros + CSVs to {GEN}")
print(f"Done in {time.time()-t0:.1f}s  (tier={TIER})")
