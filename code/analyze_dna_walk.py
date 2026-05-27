from __future__ import annotations
import gzip
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": False, "mathtext.fontset": "cm", "font.family": "serif",
    "axes.unicode_minus": False, "font.size": 9.5, "axes.labelsize": 10.5,
    "axes.titlesize": 10.0, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 8.5, "lines.linewidth": 1.6, "lines.markersize": 4.5,
    "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True, "xtick.minor.visible": True,
    "ytick.minor.visible": True, "axes.grid": True, "grid.alpha": 0.18,
    "grid.linewidth": 0.5, "legend.frameon": True, "legend.framealpha": 0.9,
    "legend.edgecolor": "0.8", "legend.fancybox": False,
    "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "axes.prop_cycle": plt.cycler(color=[
        "#0072B2", "#E69F00", "#009E73", "#D55E00",
        "#CC79A7", "#56B4E9", "#000000", "#F0E442"]),
})
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fractal_estimator import (estimate_dimension, occupancy_dimension,
                               dfa_hurst, variogram_hurst, higuchi_dimension)
try:
    import cupy as _cp  # noqa: F401
    from fractal_estimator import box_count_gpu as _box_count
    _BACKEND = "gpu"
except Exception:
    from fractal_estimator import box_count as _box_count
    _BACKEND = "cpu"

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
FIG = PAPER / "figures"; GEN = PAPER / "generated"
GENOME = PAPER / "data" / "ecoli.fna.gz"
ACCESSION = "NC_000913.3"
SEED = 2026
N_REP = 20
NS_EXP = [14, 16, 18, 20]
SCALES = 18
EPS = 0.5 ** np.arange(1, SCALES + 1)
WINDOW = (2, 7)
EST = ["box", "occ", "dfa", "vario", "higuchi"]


def graph_of(y):
    t = np.linspace(0.0, 1.0, y.size)
    yr = y - y.min(); rng = yr.max()
    yr = yr / rng if rng > 0 else yr
    return np.column_stack([t, yr])


def estimators(steps_prefix):
    y = np.cumsum(steps_prefix)
    _, c = _box_count(graph_of(y), n_scales=SCALES)
    return {"box": estimate_dimension(EPS, c, WINDOW),
            "occ": occupancy_dimension(EPS, c, n_eff=y.size),
            "dfa": 2.0 - dfa_hurst(y),
            "vario": 2.0 - variogram_hurst(y),
            "higuchi": higuchi_dimension(y)}


def transition_probs(steps):
    isR = steps > 0
    rr = int(np.sum(isR[:-1] & isR[1:])); ry = int(np.sum(isR[:-1] & ~isR[1:]))
    yr = int(np.sum(~isR[:-1] & isR[1:])); yy = int(np.sum(~isR[:-1] & ~isR[1:]))
    pRR = rr / max(rr + ry, 1); pYY = yy / max(yr + yy, 1)
    return pRR, pYY, float(isR.mean())


def dinuc_markov(N, pRR, pYY, p_start_R, rng):
    mean_run = 0.5 * (1.0 / max(1 - pRR, 1e-9) + 1.0 / max(1 - pYY, 1e-9))
    n = int(N / mean_run * 1.6) + 50
    LR = rng.geometric(max(1 - pRR, 1e-9), n)
    LY = rng.geometric(max(1 - pYY, 1e-9), n)
    lens = np.empty(2 * n, dtype=np.int64); vals = np.empty(2 * n)
    if rng.random() < p_start_R:
        lens[0::2] = LR; vals[0::2] = 1.0; lens[1::2] = LY; vals[1::2] = -1.0
    else:
        lens[0::2] = LY; vals[0::2] = -1.0; lens[1::2] = LR; vals[1::2] = 1.0
    seq = np.repeat(vals, lens)
    while seq.size < N:
        extra = np.repeat([1.0, -1.0], [rng.geometric(max(1 - pRR, 1e-9)),
                                        rng.geometric(max(1 - pYY, 1e-9))])
        seq = np.concatenate([seq, extra])
    return seq[:N]


with gzip.open(GENOME, "rt") as f:
    f.readline()
    seq = "".join(line.strip() for line in f)
arr = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
steps = np.where((arr == ord("A")) | (arr == ord("G")), 1.0, -1.0)
walk = np.cumsum(steps)
n_bases = int(arr.size)
NS = [2 ** k for k in NS_EXP if 2 ** k <= steps.size]
logN = [int(np.log2(N)) for N in NS]
print(f"[{_BACKEND}] genome {ACCESSION}: {n_bases:,} bases; {len(NS)} prefixes, "
      f"{N_REP} null replicates each.")

rng = np.random.default_rng(SEED)
real = {e: [] for e in EST}
nulls = {"mono": {e: {"mean": [], "lo": [], "hi": [], "sd": []} for e in EST},
         "dinuc": {e: {"mean": [], "lo": [], "hi": [], "sd": []} for e in EST}}
for N in NS:
    sub = steps[:N]
    r = estimators(sub)
    for e in EST:
        real[e].append(r[e])
    pRR, pYY, pR = transition_probs(sub)
    rep = {"mono": {e: [] for e in EST}, "dinuc": {e: [] for e in EST}}
    for _ in range(N_REP):
        for e, v in estimators(rng.permutation(sub)).items():
            rep["mono"][e].append(v)
        for e, v in estimators(dinuc_markov(N, pRR, pYY, pR, rng)).items():
            rep["dinuc"][e].append(v)
    for kind in ("mono", "dinuc"):
        for e in EST:
            a = np.array(rep[kind][e], dtype=float); a = a[np.isfinite(a)]
            m, sd = float(a.mean()), float(a.std(ddof=1))
            lo, hi = float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))
            nulls[kind][e]["mean"].append(m); nulls[kind][e]["sd"].append(sd)
            nulls[kind][e]["lo"].append(lo); nulls[kind][e]["hi"].append(hi)
    print(f"  N=2^{int(np.log2(N))}: real box={r['box']:.3f} dfa={r['dfa']:.3f}  "
          f"mono dfa={nulls['mono']['dfa']['mean'][-1]:.3f} "
          f"dinuc dfa={nulls['dinuc']['dfa']['mean'][-1]:.3f}")

for e in EST:
    real[e] = np.array(real[e])
    for kind in ("mono", "dinuc"):
        for q in ("mean", "lo", "hi", "sd"):
            nulls[kind][e][q] = np.array(nulls[kind][e][q])

fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(14.0, 4.2))
seg = walk[:NS[min(1, len(NS) - 1)]]
axA.plot(np.arange(seg.size), seg, lw=0.5, color="C4")
axA.set_xlabel("base index $i$"); axA.set_ylabel("DNA walk $y_i$")
axA.set_title(f"(A) E. coli DNA walk\n({ACCESSION}, first $2^{{{logN[min(1,len(NS)-1)]}}}$ bases)")

axB.plot(logN, real["box"], "ko-", lw=1.6, label="real DNA", zorder=5)
for kind, col, lab in [("mono", "C0", "mononucleotide null"),
                       ("dinuc", "C1", "dinucleotide null")]:
    m = nulls[kind]["box"]["mean"]
    axB.plot(logN, m, col + "-", lw=1.2, label=lab)
    axB.fill_between(logN, nulls[kind]["box"]["lo"], nulls[kind]["box"]["hi"],
                     color=col, alpha=0.20)
axB.set_xlabel(r"$\log_2 N$"); axB.set_ylabel("box-counting graph dimension")
axB.set_title("(B) Box-counting drifts with $N$;\nreal vs. null bands (2.5--97.5\\% quantiles)")
axB.legend(fontsize=8)

for e, col, mk, lab in [("box", "C2", "o", "box-counting"),
                        ("dfa", "C3", "^", "DFA $2-H$"),
                        ("vario", "C5", "v", "variogram $2-H$")]:
    gap = real[e] - nulls["mono"][e]["mean"]
    err = np.vstack([nulls["mono"][e]["mean"] - nulls["mono"][e]["lo"],
                     nulls["mono"][e]["hi"] - nulls["mono"][e]["mean"]])
    axC.errorbar(logN, gap, yerr=err, marker=mk, color=col, capsize=3, lw=1.2, label=lab)
axC.axhline(0.0, color="gray", ls="--", lw=1.0)
axC.set_xlabel(r"$\log_2 N$")
axC.set_ylabel("real $-$ mononucleotide null (dim.)")
axC.set_title("(C) Biological gap vs. null\n(DFA/variogram resolve it; box-counting noisier)")
axC.legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "fig_realdata.png"); plt.close(fig)

j = len(NS) - 1
def row(name, get):
    return (f"{name} & ${get('box')}$ & ${get('occ')}$ & ${get('dfa')}$ & "
            f"${get('vario')}$ & ${get('higuchi')}$ & ${get('H')}$ & ${get('drift')}$\\\\")

def real_get(k):
    if k == "H": return f"{2.0 - real['dfa'][j]:.2f}"
    if k == "drift": return f"{real['box'][j] - real['box'][0]:+.2f}"
    return f"{real[k][j]:.2f}"

def null_get(kind):
    def g(k):
        if k == "H": return f"{2.0 - nulls[kind]['dfa']['mean'][j]:.2f}"
        if k == "drift":
            return f"{nulls[kind]['box']['mean'][j] - nulls[kind]['box']['mean'][0]:+.2f}"
        return f"{nulls[kind][k]['mean'][j]:.2f}"
    return g

tlines = [
    r"\begin{table}[t]", r"\centering",
    r"\caption{Real \emph{E. coli} DNA walk versus finite-size null surrogates at"
    f" $N=2^{{{logN[j]}}}$ (nulls: mean over {N_REP} replicates). The mononucleotide"
    r" shuffle preserves composition; the dinucleotide (first-order Markov)"
    r" surrogate matches the lag-1 statistics in expectation. Box-counting drifts with $N$ and"
    r" cannot be read as a fixed dimension; the structure-function estimators are"
    r" stable. The real walk's Hurst exponent exceeds both null bands, consistent"
    r" with long-range correlation beyond composition and lag-1 dependence under"
    r" these null models; the true"
    r" dimension is unknown, so this is an empirical comparison, not a validation"
    r" against known truth.}",
    r"\label{tab:dna}", r"\small",
    r"\begin{tabular}{lccccccc}", r"\toprule",
    r"sequence & box & occ.-fit & DFA & variogram & Higuchi & $H$ & box drift\\",
    r"\midrule",
    row(r"real DNA", real_get),
    row(r"mononucleotide null", null_get("mono")),
    row(r"dinucleotide null", null_get("dinuc")),
    r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
]
(GEN / "results_dna.tex").write_text("\n".join(tlines), encoding="utf-8")

realH = 2.0 - real["dfa"][j]
monoH = 2.0 - nulls["mono"]["dfa"]["mean"][j]
monoH_lo = 2.0 - nulls["mono"]["dfa"]["hi"][j]
monoH_hi = 2.0 - nulls["mono"]["dfa"]["lo"][j]
dinucH = 2.0 - nulls["dinuc"]["dfa"]["mean"][j]
dinucH_lo = 2.0 - nulls["dinuc"]["dfa"]["hi"][j]
dinucH_hi = 2.0 - nulls["dinuc"]["dfa"]["lo"][j]

def mc(name, val):
    return f"\\newcommand{{\\{name}}}{{{val}}}"

mlines = [
    "% Auto-generated by analyze_dna_walk.py -- do not edit by hand.",
    mc("dnaAcc", r"NC\_000913.3"), mc("dnaBases", f"{n_bases:,}"),
    mc("dnaNrep", str(N_REP)), mc("dnaNlo", str(logN[0])), mc("dnaNhi", str(logN[j])),
    mc("dnaRealBox", f"{real['box'][j]:.2f}"), mc("dnaRealOcc", f"{real['occ'][j]:.2f}"),
    mc("dnaRealDFA", f"{real['dfa'][j]:.2f}"), mc("dnaRealVario", f"{real['vario'][j]:.2f}"),
    mc("dnaRealH", f"{realH:.2f}"),
    mc("dnaBoxDrift", f"{abs(real['box'][j] - real['box'][0]):.2f}"),
    mc("dnaMonoH", f"{monoH:.2f}"), mc("dnaMonoHlo", f"{monoH_lo:.2f}"), mc("dnaMonoHhi", f"{monoH_hi:.2f}"),
    mc("dnaDinucH", f"{dinucH:.2f}"), mc("dnaDinucHlo", f"{dinucH_lo:.2f}"), mc("dnaDinucHhi", f"{dinucH_hi:.2f}"),
    mc("dnaHexcess", f"{realH - monoH:.2f}"),
]
(GEN / "macros_dna.tex").write_text("\n".join(mlines) + "\n", encoding="utf-8")
print(f"real H={realH:.2f}  mono H={monoH:.2f} [{monoH_lo:.2f},{monoH_hi:.2f}]  "
      f"dinuc H={dinucH:.2f}  excess={realH-monoH:+.2f}")
print("wrote figures/fig_realdata.png, generated/results_dna.tex, generated/macros_dna.tex")
