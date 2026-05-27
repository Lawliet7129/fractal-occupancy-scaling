# Finite-size occupancy scaling of apparent fractal dimensions in stochastic trajectories

Reproducibility code for the paper *"Finite-size occupancy scaling of apparent
fractal dimensions in stochastic trajectories"* (Bon A. Koo and Edward Ju). The
pipeline regenerates every figure, table, and in-text number from a single
master seed.

**Central result.** The box-counting bias of a finite stochastic trajectory is a
finite-size occupancy crossover. A balls-in-boxes model gives the saturation
scale `k* = (1/Df) log2(Neff/A)`, predicts the full box-count curve, and
collapses the normalized local slope of walk traces (`Df = 2`), fractional
Brownian graphs (`Df = 2 - H`), and Lévy flights (`Df = alpha`) onto one
parameter-free crossover curve. Inverting the model yields a finite-size bias
correction, validated out of class and illustrated on an *E. coli* DNA walk.

## Layout
| Path | Purpose |
|---|---|
| `code/fractal_estimator.py` | process generators (SRW, Pearson, Brownian, persistent / anti-persistent walks, Lévy flight, fBm graph) + estimators (box-counting, correlation dimension, DFA, variogram, Higuchi) + the occupancy model and finite-size-scaling helpers |
| `code/generate_paper_assets.py` | runs the full simulation grid; writes all figures, tables, CSVs, and macros |
| `code/analyze_dna_walk.py` | empirical application: *E. coli* DNA walk vs. mononucleotide / dinucleotide null surrogates |
| `code/analyze_nondyadic.py` | non-dyadic grid-base robustness check |
| `code/build_robustness_table.py` | default- vs. full-tier robustness comparison table |
| `data/ecoli.fna.gz` | bundled real genome (NCBI RefSeq NC_000913.3), the only non-simulated input |
| `figures/` | generated figures, committed as reference outputs |
| `generated/` | raw numerical outputs (`*.csv`): the per-cell results, collapse scatter, and estimator sweeps behind the figures and tables |

## Install
```bash
pip install -r requirements.txt
```
Python 3.9+. Core dependencies are NumPy, SciPy, and Matplotlib. An optional
CuPy GPU backend (`--gpu`) accelerates the box count ~40x and returns the *same*
integer counts as the CPU path.

## Reproduce
```bash
cd code
python generate_paper_assets.py --full        # full tier (N up to 2^20, d up to 20), CPU
python generate_paper_assets.py --full --gpu   # identical numbers on GPU (~1-1.5 h vs ~12 h)
python generate_paper_assets.py                 # default tier (N up to 2^16, d up to 10)
python generate_paper_assets.py --fast          # smoke test (< 1 min)
python analyze_dna_walk.py                       # real-data DNA walk (reads ../data/ecoli.fna.gz)
python analyze_nondyadic.py                      # non-dyadic grid robustness
python build_robustness_table.py                 # robustness comparison table
```
Scripts resolve input/output paths relative to the repository root, so keep the
`code / data / figures / generated` layout intact. A single master seed (2026)
controls all randomness, so reruns reproduce the committed outputs. The
committed `figures/` and `generated/*.csv` outputs are the **full tier**.

To re-fetch the genome:
```bash
curl -sL -o data/ecoli.fna.gz "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/005/845/GCF_000005845.2_ASM584v2/GCF_000005845.2_ASM584v2_genomic.fna.gz"
```

## Notes
- The box count `N(eps)` is computed once per `(model, N, d, seed)` and cached;
  every window-dependent quantity (slopes, stability `S(W)`, the rescaled
  coordinate `x`) is derived from that cache.
- Known true dimensions (the targets the estimators are checked against): walk
  trace = 2 for all `d >= 2` (Taylor 1953); fBm graph = `2 - H`; Lévy-flight
  range = `alpha`.
- The pipeline also writes LaTeX table fragments (`results_*.tex`) and
  `\newcommand` macro files (`macros*.tex`) into `generated/` for the manuscript.
  Those are git-ignored here — they live with the paper, not this code repo — so
  only the numerical `*.csv` outputs are tracked.

## License
MIT — see [LICENSE](LICENSE).
