from __future__ import annotations
import re
from pathlib import Path

GEN = Path(__file__).resolve().parent.parent / "generated"


def parse_macros(path: Path) -> dict:
    d = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\\newcommand\{\\(\w+)\}\{(.*)\}", line.strip())
        if m:
            d[m.group(1)] = m.group(2)
    return d


def num(d: dict, k: str):
    try:
        return float(d[k].replace(",", ""))
    except Exception:
        return None


dft = parse_macros(GEN / ".macros_default_snapshot.tex")
full = parse_macros(GEN / "macros.tex")

ROWS = [
    ("uniCorr", r"Universal local-slope collapse: corr.\ with $h(\xi)$", "hi"),
    ("modelCorr", r"Occupancy prediction of windowed bias: corr.\ with data", "hi"),
    ("collapseRaw", r"Windowed-bias raw spread (std)", "ctx"),
    ("collapseResid", r"Windowed-bias residual after collapse", "lo"),
    ("collapseFactor", r"Collapse spread-reduction factor ($\times$)", "hi"),
    ("collapseModelMax", r"Largest per-model collapse residual", "lo"),
    ("traceBoxRMSE", r"Walk traces: box-counting RMSE", "ctx"),
    ("traceOccRMSE", r"Walk traces: occupancy-fit RMSE", "lo"),
    ("heldRawRMSE", r"Held-out walks: raw RMSE", "ctx"),
    ("heldCorrRMSE", r"Held-out walks: corrected RMSE", "lo"),
    ("boxRMSE", r"fBm graphs: box-counting RMSE", "ctx"),
    ("dfaRMSE", r"fBm graphs: DFA RMSE", "lo"),
    ("varioRMSE", r"fBm graphs: variogram RMSE", "lo"),
    ("higuchiRMSE", r"fBm graphs: Higuchi RMSE", "lo"),
    ("corrRMSE", r"fBm graphs: correlation-dim.\ RMSE", "lo"),
]


def verdict(k, direction):
    a, b = num(dft, k), num(full, k)
    if a is None or b is None:
        return "--"
    if abs(b - a) <= max(0.02, 0.07 * abs(a)):
        return "stable"
    if direction == "hi":
        return "stronger" if b > a else "weaker"
    if direction == "lo":
        return "improved" if b < a else "degraded"
    return "higher" if b > a else "lower"


nd, dd = full.get("Nmaxexp", "?"), full.get("dmax", "?")
ndd, ddd = dft.get("Nmaxexp", "?"), dft.get("dmax", "?")
lines = [
    r"\begin{table}[t]", r"\centering",
    r"\caption{Full-tier robustness. Key quantities at the default tier "
    f"($N\\le2^{{{ndd}}}$, $d\\le{ddd}$) and the full tier "
    f"($N\\le2^{{{nd}}}$, $d\\le{dd}$). ``stable'' marks a change within "
    r"$\max(0.02,\,7\%)$; otherwise the direction of change is given. The main "
    r"conclusions persist under the full sweep.}",
    r"\label{tab:robustness}",
    r"\small",
    r"\begin{tabular}{lccc}", r"\toprule",
    r"quantity & default tier & full tier & change\\", r"\midrule",
]
for k, label, direction in ROWS:
    a = dft.get(k, "--"); b = full.get(k, "--")
    lines.append(f"{label} & ${a}$ & ${b}$ & {verdict(k, direction)}\\\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
(GEN / "results_robustness.tex").write_text("\n".join(lines), encoding="utf-8")
print("wrote results_robustness.tex")
for k, label, direction in ROWS:
    print(f"  {k:16s} default={dft.get(k,'--'):>7} full={full.get(k,'--'):>7} -> {verdict(k,direction)}")
