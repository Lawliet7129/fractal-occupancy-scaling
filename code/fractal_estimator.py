from __future__ import annotations
import numpy as np

TRUE_TRACE_DIM = 2.0


def simple_random_walk(n_steps, dim, rng):
    axes = rng.integers(0, dim, size=n_steps)
    signs = rng.choice(np.array([-1.0, 1.0]), size=n_steps)
    steps = np.zeros((n_steps, dim), dtype=np.float64)
    steps[np.arange(n_steps), axes] = signs
    verts = np.empty((n_steps + 1, dim), dtype=np.float64)
    verts[0] = 0.0
    np.cumsum(steps, axis=0, out=verts[1:])
    return verts


def pearson_walk(n_steps, dim, rng):
    g = rng.standard_normal((n_steps, dim))
    norms = np.linalg.norm(g, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    g /= norms
    verts = np.empty((n_steps + 1, dim), dtype=np.float64)
    verts[0] = 0.0
    np.cumsum(g, axis=0, out=verts[1:])
    return verts


def brownian_path(n_steps, dim, rng):
    g = rng.standard_normal((n_steps, dim))
    verts = np.empty((n_steps + 1, dim), dtype=np.float64)
    verts[0] = 0.0
    np.cumsum(g, axis=0, out=verts[1:])
    return verts


def correlated_walk(n_steps, dim, rng, rho):
    from scipy.signal import lfilter
    z = rng.standard_normal((n_steps, dim))
    c = np.sqrt(1.0 - rho * rho)
    zi = (z[0] * (1.0 - c))[None, :]
    e, _ = lfilter([c], [1.0, -rho], z, axis=0, zi=zi)
    verts = np.empty((n_steps + 1, dim), dtype=np.float64)
    verts[0] = 0.0
    np.cumsum(e, axis=0, out=verts[1:])
    return verts


def persistent_walk(n_steps, dim, rng, rho=0.5):
    return correlated_walk(n_steps, dim, rng, rho)


def antipersistent_walk(n_steps, dim, rng, rho=-0.5):
    return correlated_walk(n_steps, dim, rng, rho)


def fbm_graph(n_steps, H, rng):
    k = np.arange(n_steps + 1)
    g = 0.5 * (np.abs(k - 1) ** (2 * H) - 2 * np.abs(k) ** (2 * H) + np.abs(k + 1) ** (2 * H))
    c = np.concatenate([g, g[-2:0:-1]])
    lam = np.clip(np.fft.fft(c).real, 0.0, None)
    V = rng.standard_normal(len(c)) + 1j * rng.standard_normal(len(c))
    fgn = np.fft.fft(np.sqrt(lam) * V).real[:n_steps] / np.sqrt(len(c))
    y = np.concatenate([[0.0], np.cumsum(fgn)]) * (n_steps ** (-H))
    return np.column_stack([np.linspace(0.0, 1.0, n_steps + 1), y])


def levy_flight(n_steps, dim, rng, alpha):
    U = rng.uniform(-np.pi / 2, np.pi / 2, size=(n_steps, dim))
    W = rng.exponential(1.0, size=(n_steps, dim))
    if abs(alpha - 1.0) < 1e-9:
        X = np.tan(U)
    else:
        X = (np.sin(alpha * U) / np.power(np.cos(U), 1.0 / alpha)) * \
            np.power(np.cos((1.0 - alpha) * U) / W, (1.0 - alpha) / alpha)
    verts = np.empty((n_steps + 1, dim), dtype=np.float64)
    verts[0] = 0.0
    np.cumsum(X, axis=0, out=verts[1:])
    return verts


TRACE_MODELS = {
    "SRW": simple_random_walk,
    "Pearson": pearson_walk,
    "Brownian": brownian_path,
    "Persistent": persistent_walk,
    "Antipersistent": antipersistent_walk,
}


def _count_unique_rows(cells):
    cells = np.ascontiguousarray(cells)
    void_view = cells.view(np.dtype((np.void, cells.dtype.itemsize * cells.shape[1])))
    return int(np.unique(void_view).shape[0])


def box_count(points, n_scales=12, n_offsets=1, rng=None):
    mn = points.min(axis=0)
    extent = (points.max(axis=0) - mn).max()
    if extent <= 0:
        extent = 1.0
    P = (points - mn) / extent
    eps = 0.5 ** np.arange(1, n_scales + 1)
    counts = np.empty(n_scales, dtype=np.float64)
    for i, e in enumerate(eps):
        if n_offsets == 1:
            m = int(round(1.0 / e))
            counts[i] = _count_unique_rows(np.minimum(np.floor(P / e).astype(np.int64), m - 1))
        else:
            counts[i] = np.mean([_count_unique_rows(
                np.floor((P + rng.uniform(0.0, e, P.shape[1])) / e).astype(np.int64))
                for _ in range(n_offsets)])
    return eps, counts


def box_count_gpu(points, n_scales=12, n_offsets=1, rng=None):
    import cupy as cp
    R = np.uint64(1099511628211)
    OFFB = np.uint64(1469598103934665603)
    P0 = cp.asarray(points, dtype=cp.float64)
    mn = P0.min(axis=0)
    extent = float((P0.max(axis=0) - mn).max())
    if extent <= 0.0:
        extent = 1.0
    P = (P0 - mn) / extent
    npts = int(P.shape[0]); dim = int(P.shape[1])
    eps = 0.5 ** np.arange(1, n_scales + 1)
    counts = np.empty(n_scales, dtype=np.float64)

    def n_unique_rows(cells):
        key = cp.full(npts, OFFB, dtype=cp.uint64)
        for j in range(dim):
            key = (key ^ cells[:, j].astype(cp.uint64)) * R
        return int(cp.unique(key).shape[0])

    saturated = False
    for i in range(n_scales):
        if saturated:
            counts[i] = npts
            continue
        e = 0.5 ** (i + 1)
        if n_offsets == 1:
            m = 1 << (i + 1)
            cells = cp.minimum(cp.floor(P / e).astype(cp.int64), m - 1)
            cnt = n_unique_rows(cells)
            counts[i] = cnt
            if cnt >= npts:
                saturated = True
        else:
            vals = []
            for _ in range(n_offsets):
                off = cp.asarray(rng.uniform(0.0, e, dim))
                cells = cp.floor((P + off) / e).astype(cp.int64)
                vals.append(n_unique_rows(cells))
            counts[i] = float(np.mean(vals))
    del P0, P
    cp.get_default_memory_pool().free_all_blocks()
    return eps, counts


def box_count_base(points, base, n_scales):
    mn = points.min(axis=0)
    extent = (points.max(axis=0) - mn).max()
    if extent <= 0:
        extent = 1.0
    P = (points - mn) / extent
    eps = base ** (-np.arange(1, n_scales + 1, dtype=np.float64))
    counts = np.empty(n_scales, dtype=np.float64)
    for i, e in enumerate(eps):
        m = int(np.ceil(1.0 / e))
        counts[i] = _count_unique_rows(np.minimum(np.floor(P / e).astype(np.int64), m - 1))
    return eps, counts


def estimate_dimension(eps, counts, window):
    x = np.log(1.0 / eps)
    y = np.log(counts.astype(np.float64))
    lo, hi = window
    sel = slice(lo - 1, hi)
    slope, _ = np.polyfit(x[sel], y[sel], 1)
    return float(slope)


def local_slopes(counts):
    lc = np.log2(np.asarray(counts, dtype=np.float64))
    return np.diff(lc)


def window_stability(counts, window):
    lo, hi = window
    s = local_slopes(counts)
    seg = s[lo - 1:hi - 1]
    if seg.size < 2:
        return float("nan")
    return float(np.std(seg, ddof=1))


def k_star(dim, n_points):
    return 0.5 * np.log2(n_points / dim)


def rescaled_coord(window, dim, n_points):
    lo, hi = window
    return 0.5 * (lo + hi) - k_star(dim, n_points)


def occupancy_logcount(k, n_points, Df, A):
    b = np.maximum(A * np.power(2.0, Df * np.asarray(k, dtype=np.float64)), 1.0 + 1e-9)
    frac = -np.expm1(n_points * np.log1p(-1.0 / b))
    return np.log(np.maximum(b * frac, 1e-300))


def occupancy_universal_slope(xi):
    u = np.power(2.0, np.asarray(xi, dtype=np.float64))
    e = np.exp(-1.0 / u)
    return 1.0 - e / (u * (1.0 - e))


def occupancy_slope_fd(k, n_eff, Df, A):
    y0 = occupancy_logcount(k, n_eff, Df, A) / np.log(2.0)
    y1 = occupancy_logcount(k + 1, n_eff, Df, A) / np.log(2.0)
    return (y1 - y0) / Df


def effective_n(points):
    return _count_unique_rows(np.asarray(points))


def k_star_model(n_points, Df, A):
    return (1.0 / Df) * np.log2(n_points / A)


def fit_occupancy(eps, counts, n_eff=None, Df=None):
    from scipy.optimize import curve_fit
    counts = np.asarray(counts, dtype=np.float64)
    if n_eff is None:
        n_eff = float(counts.max())
    k = np.log2(1.0 / np.asarray(eps, dtype=np.float64))
    y = np.log(counts)
    m = counts >= 2
    if m.sum() < 4:
        m = np.ones_like(k, dtype=bool)
    if Df is None:
        p, _ = curve_fit(lambda kk, D, A: occupancy_logcount(kk, n_eff, D, A),
                         k[m], y[m], p0=[1.5, 1.0],
                         bounds=([0.2, 1e-3], [3.0, 1e6]), maxfev=40000)
        return float(p[0]), float(p[1])
    p, _ = curve_fit(lambda kk, A: occupancy_logcount(kk, n_eff, Df, A),
                     k[m], y[m], p0=[1.0], bounds=(1e-3, 1e6), maxfev=20000)
    return float(Df), float(p[0])


def occupancy_dimension(eps, counts, n_eff=None):
    try:
        return fit_occupancy(eps, counts, n_eff=n_eff, Df=None)[0]
    except Exception:
        return float("nan")


def model_windowed_bias(window, n_points, Df, A):
    lo, hi = window
    k = np.arange(lo, hi + 1)
    y = occupancy_logcount(k, n_points, Df, A)
    x = np.log(2.0) * k
    return float(np.polyfit(x, y, 1)[0] - Df)


def correlation_dimension(points, rng, n_sub=1500, n_radii=24, frac=(0.15, 0.55)):
    from scipy.spatial.distance import pdist
    P = points
    if len(P) > n_sub:
        idx = rng.choice(len(P), size=n_sub, replace=False)
        P = P[idx]
    d = pdist(P)
    d = d[d > 0]
    if d.size < 10:
        return float("nan")
    radii = np.logspace(np.log10(d.min()), np.log10(d.max()), n_radii)
    C = np.array([np.mean(d < r) for r in radii])
    ok = C > 0
    x, y = np.log(radii[ok]), np.log(C[ok])
    lo, hi = int(len(x) * frac[0]), int(len(x) * frac[1])
    if hi - lo < 2:
        lo, hi = 0, len(x)
    slope, _ = np.polyfit(x[lo:hi], y[lo:hi], 1)
    return float(slope)


def correlation_dimension_theiler(points, rng, n_sub=2000, theiler=0, n_radii=24, frac=(0.15, 0.55)):
    from scipy.spatial.distance import pdist, squareform
    n = len(points)
    idx = np.sort(rng.choice(n, size=n_sub, replace=False)) if n > n_sub else np.arange(n)
    P = points[idx]; m = len(P)
    Dm = squareform(pdist(P))
    tri = np.arange(m)[:, None] < np.arange(m)[None, :]
    keep = tri & (np.abs(idx[:, None] - idx[None, :]) > theiler)
    d = Dm[keep]; d = d[d > 0]
    if d.size < 10:
        return float("nan")
    radii = np.logspace(np.log10(d.min()), np.log10(d.max()), n_radii)
    C = np.array([np.mean(d < r) for r in radii])
    ok = C > 0
    x, y = np.log(radii[ok]), np.log(C[ok])
    lo, hi = int(len(x) * frac[0]), int(len(x) * frac[1])
    if hi - lo < 2:
        lo, hi = 0, len(x)
    slope, _ = np.polyfit(x[lo:hi], y[lo:hi], 1)
    return float(slope)


def _series_from_graph(graph_or_series):
    a = np.asarray(graph_or_series, dtype=np.float64)
    return a[:, -1] if a.ndim == 2 else a


def dfa_hurst(graph_or_series, scales=None, order=1):
    y = _series_from_graph(graph_or_series)
    incr = np.diff(y)
    n = incr.size
    if n < 32:
        return float("nan")
    profile = np.cumsum(incr - incr.mean())
    if scales is None:
        smax = n // 4
        scales = np.unique(np.floor(np.logspace(
            np.log10(8), np.log10(max(8, smax)), 16)).astype(int))
        scales = scales[scales >= order + 2]
    F, used = [], []
    for s in scales:
        nseg = n // s
        if nseg < 1:
            continue
        seg = profile[:nseg * s].reshape(nseg, s)
        t = np.arange(s)
        rms = []
        for row in seg:
            coef = np.polyfit(t, row, order)
            resid = row - np.polyval(coef, t)
            rms.append(np.mean(resid ** 2))
        F.append(np.sqrt(np.mean(rms)))
        used.append(s)
    F = np.asarray(F)
    used = np.asarray(used, dtype=np.float64)
    ok = F > 0
    if ok.sum() < 3:
        return float("nan")
    slope, _ = np.polyfit(np.log(used[ok]), np.log(F[ok]), 1)
    return float(slope)


def variogram_hurst(graph_or_series, lags=None):
    y = _series_from_graph(graph_or_series)
    n = y.size
    if n < 16:
        return float("nan")
    if lags is None:
        lags = np.unique(np.floor(np.logspace(0, np.log10(n // 4), 16)).astype(int))
        lags = lags[lags >= 1]
    V, used = [], []
    for d in lags:
        diff = y[d:] - y[:-d]
        if diff.size:
            V.append(np.mean(diff ** 2))
            used.append(d)
    V = np.asarray(V)
    used = np.asarray(used, dtype=np.float64)
    ok = V > 0
    if ok.sum() < 3:
        return float("nan")
    slope, _ = np.polyfit(np.log(used[ok]), np.log(V[ok]), 1)
    return float(slope / 2.0)


def higuchi_dimension(graph_or_series, kmax=32):
    y = _series_from_graph(graph_or_series)
    n = y.size
    if n < 2 * kmax:
        kmax = max(2, n // 4)
    L, used = [], []
    for k in range(1, kmax + 1):
        Lk = []
        for m in range(k):
            idx = np.arange(m, n, k)
            if idx.size < 2:
                continue
            seg = y[idx]
            length = np.sum(np.abs(np.diff(seg)))
            norm = (n - 1) / ((idx.size - 1) * k)
            Lk.append(length * norm / k)
        if Lk:
            L.append(np.mean(Lk))
            used.append(k)
    L = np.asarray(L)
    used = np.asarray(used, dtype=np.float64)
    ok = L > 0
    if ok.sum() < 3:
        return float("nan")
    slope, _ = np.polyfit(np.log(1.0 / used[ok]), np.log(L[ok]), 1)
    return float(slope)


def bootstrap_ci(samples, n_boot=2000, seed=12345):
    rng = np.random.default_rng(seed)
    samples = np.asarray(samples, dtype=np.float64)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, len(samples), size=(n_boot, len(samples)))
    means = samples[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def rmse(values, truth):
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((v - truth) ** 2)))
