# h001 simulation — reproduction package

- **Repository:** https://github.com/sun0720/005_gls
- **DOI:** [CONFIRM] archive DOI once the release is minted
- **License:** MIT — see `LICENSE`
- **Environment:** Python 3.11; `requirements.txt` is authoritative where the description below is less specific

## Contents

- **Scripts (entry points):** `sim.py`
- **Outputs:** `results/`

## Package description

Run from the repository root:

```bash
python3 sim.py
```

The script writes every table and figure to `results/` next to `sim.py`.
It reads no external data and uses only NumPy, SciPy, and Matplotlib (versions in
`requirements.txt`).

## Master seed and stream allocation

- Master seed: `20261022`.
- Main Monte Carlo (45 cells: `n ∈ {25,50,100,200,400}` × `c ∈ {0.05,0.10,0.20}`
  × `a ∈ {0,c/2,c}`): `SeedSequence(20261022).spawn(15)`, one child per
  `(n,c)` cell.
- Conditional-variance check (10 fixed regressors × 300 error draws):
  `20261022 + 7 n + int(1000 c)`.
- D3 local expansion: deterministic, no RNG.
- D4 minimax: deterministic, no RNG.
- D8 pointwise panel: `20261022 + 7300 + 17 i + j` (`i` indexes `n ∈ {100,200}`, `j` indexes `c`).
- Minimax coverage panel: `20261022 + 9100 + 31 i + j`.
- Negative case (independence violation): `20261022 + 91`.
- Single draw (§6): `20261022 + 777 = 20261799`.
- OLS benchmark: `20261022 + 3000 + n + int(1000 c)`.
- OLS-HAC comparator (§5.10): the same stream as the OLS benchmark, so the two panels are cell-by-cell comparable; bandwidths `b = floor(n/4)` and `b = 5`.
- GLS-HAC comparator (§5.10): the same stream, with the GLS weight at `a=0`; bandwidths `b = 5` and `b = floor(n/4)`, `t_{n-1}` critical values.
- Lowest-coverage recheck: `20261022 + 405`, `R=5000`, pre-announced rule.
- Strong-persistence cells: `20261022 + 5150` and `+ 5151`.
- Truth-outside-`C` boundary cells: `20261022 + 5250` and `+ 5251`.

## Outputs

| File | Content |
|---|---|
| `results.csv` | 45 main cells: bias, RMSE, coverage, mean variances |
| `d1.csv` | covariance-expansion error (spectral norm / spectral norm) |
| `d2.csv` | conditional-variance check |
| `d3.csv` / `d4.csv` | local expansion / minimax grid |
| `d8.csv` | pointwise interval, both signs of the gap |
| `hac.csv` | feasible comparator: OLS with a spatial HAC variance, two bandwidths |
| `gls_hac.csv` | feasible comparator: GLS at `a=0` with a spatial HAC variance, two bandwidths, plus the scale diagnostic (`coverage_true_sd`, `mean_se_over_rmse`, `t_skew`, `t_exkurt`) |
| `negative.csv` | independence violation |
| `ols.csv`, `single_draw.csv`, `coverage_recheck.csv` | benchmarks |
| `minimax.csv` | minimax tuning coverage |
| `strong_persistence.csv` | two interior strong-persistence cells |
| `misspecified_c.csv` | boundary cells with the truth outside `C` |
| `summary.json` | machine-readable summary of every table |
| `rate.png`, `bias.png`, `coverage.png`, `local_expansion.png`, `qq.png` | figures |

## Scope note

The design is a fixed `c`, equally spaced line; it is a finite-sample validation
of the fixed-`c` formulas, not a drifting local-to-unity sequence. The seven-point
supremum equals the continuous supremum for the exponential kernel if Lemma L4
holds (monotonicity of `c ↦ V_n(a,c;x)`); the script uses the grid directly.

## Citation

If you use this material, cite the paper and this archive ([DOI to be added]).
