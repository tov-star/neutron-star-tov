"""
pulsar_population.py  --  Part 2: neutron-star (pulsar) rotation population.

Produces the spin-period histogram and TWO characterisations of the rotation
period that feed the model's initial spin P0 (Leung: "average or binned"):

  1. AVERAGE   - mean, median, and the rotation-relevant harmonic mean
                 (= 1/<1/P>, which weights the fast rotators the spin-up
                 mechanism actually cares about; for a period distribution
                 skewed to long P, harmonic mean < arithmetic mean).
  2. BINNED    - the period distribution split into log-spaced bins, each with a
                 representative period (the in-bin median) and a population
                 weight. Saved to results/period_bins.csv so the survey can be
                 run per bin and the outcomes weighted by the real population.

Because the ejection ratio scales as 1/P0, the binned table lets you turn the
single survey grid into a population-weighted ejection prediction without
re-running anything.

GET THE DATA (once, on a networked machine)
  https://www.atnf.csiro.au/research/pulsar/psrcat/  -> tick P0 -> save CSV.
  Then:  python pulsar_population.py psrcat_P0.csv  [--pop msp|all] [--nbins 12]
"""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import sys, re, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MSP_CUTOFF = 30e-3     # s


def load_periods(path, col=None):
    try:
        import pandas as pd
        df = pd.read_csv(path, sep=None, engine="python", comment="#")
        if col and col in df.columns:
            p = pd.to_numeric(df[col], errors="coerce").to_numpy()
        else:
            cand = [c for c in df.columns
                    if re.search(r"\bp0\b|period", str(c), re.I)]
            p = pd.to_numeric(df[cand[0]], errors="coerce").to_numpy() if cand else None
        if p is not None:
            p = p[np.isfinite(p)]; p = p[(p > 1e-3) & (p < 100)]
            if p.size:
                return p
    except Exception:
        pass
    periods = []
    with open(path) as f:
        for line in f:
            if line.lstrip().startswith("#"):
                continue
            for tok in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line):
                try:
                    v = float(tok)
                except ValueError:
                    continue
                if 1e-3 < v < 100:
                    periods.append(v); break
    return np.array(periods, dtype=float)


def summarize(p):
    msp = p[p < MSP_CUTOFF]
    print(f"total pulsars                : {p.size}")
    print(f"  millisecond (P<30 ms)      : {msp.size}")
    print(f"mean   period (all)          : {np.mean(p)*1e3:.2f} ms")
    print(f"median period (all)          : {np.median(p)*1e3:.2f} ms")
    if msp.size:
        hmean = 1.0 / np.mean(1.0 / msp)        # rotation-relevant average
        print(f"mean   period (MSPs)         : {np.mean(msp)*1e3:.3f} ms")
        print(f"median period (MSPs)         : {np.median(msp)*1e3:.3f} ms")
        print(f"harmonic mean (MSPs)         : {hmean*1e3:.3f} ms   <- P0 from <1/P>")
        print(f"fastest pulsar               : {np.min(p)*1e3:.3f} ms")
    return msp


def binned_periods(p, n_bins=12, pop="msp"):
    """Log-spaced period bins. Returns list of dicts with the in-bin median
    period (representative P0) and the population weight."""
    sel = p[p < MSP_CUTOFF] if pop == "msp" else p
    if sel.size == 0:
        return []
    edges = np.logspace(np.log10(sel.min()), np.log10(sel.max() * 1.001), n_bins + 1)
    rows = []
    for k in range(n_bins):
        lo, hi = edges[k], edges[k + 1]
        inb = sel[(sel >= lo) & (sel < hi)]
        if inb.size == 0:
            continue
        rows.append(dict(P_lo_ms=lo*1e3, P_hi_ms=hi*1e3,
                         P_rep_ms=float(np.median(inb)*1e3),
                         count=int(inb.size), weight=float(inb.size/sel.size)))
    return rows


def save_bins(rows, path="results/period_bins.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("P_lo_ms,P_hi_ms,P_rep_ms,count,weight\n")
        for r in rows:
            f.write(f"{r['P_lo_ms']:.4f},{r['P_hi_ms']:.4f},"
                    f"{r['P_rep_ms']:.4f},{r['count']},{r['weight']:.5f}\n")
    print(f"saved {path}")


def print_bins(rows, pop):
    print(f"\nBinned rotation period ({pop} population):")
    print(f"  {'P range [ms]':>18}{'P_rep [ms]':>12}{'count':>8}{'weight':>9}")
    for r in rows:
        print(f"  {r['P_lo_ms']:7.3f}-{r['P_hi_ms']:<8.3f}"
              f"{r['P_rep_ms']:12.3f}{r['count']:8d}{r['weight']:9.3f}")


def plot_population(p, rows, pop, outpath="results/pulsar_period_hist.png"):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    logp = np.log10(p)
    ax1.hist(logp, bins=60, color="steelblue", edgecolor="white", alpha=0.85)
    ax1.axvline(np.log10(MSP_CUTOFF), color="crimson", ls="--", label="MSP cut (30 ms)")
    ax1.axvline(np.log10(np.median(p)), color="k", ls=":",
                label=f"median {np.median(p)*1e3:.0f} ms")
    ax1.set_xticks(np.log10([1e-3, 1e-2, 1e-1, 1e0, 1e1]))
    ax1.set_xticklabels(["1 ms", "10 ms", "0.1 s", "1 s", "10 s"])
    ax1.set_xlabel(r"$\log_{10}(P/\mathrm{s})$"); ax1.set_ylabel("count")
    ax1.set_title("Spin-period distribution"); ax1.legend()
    if rows:
        reps = [r["P_rep_ms"] for r in rows]
        wts = [r["weight"] for r in rows]
        ax2.bar(range(len(reps)), wts, color="seagreen", alpha=0.85)
        ax2.set_xticks(range(len(reps)))
        ax2.set_xticklabels([f"{x:.1f}" for x in reps], rotation=45, fontsize=8)
        ax2.set_xlabel("representative period per bin [ms]")
        ax2.set_ylabel("population weight")
        ax2.set_title(f"Binned rotation period ({pop})")
    fig.tight_layout(); fig.savefig(outpath, dpi=130); plt.close(fig)
    print(f"saved {outpath}")


def main(path, pop="msp", nbins=12, col=None):
    p = load_periods(path, col=col)
    if p.size == 0:
        print("No periods parsed. Check the file, or pass --col <name>."); return
    summarize(p)
    rows = binned_periods(p, n_bins=nbins, pop=pop)
    print_bins(rows, pop)
    save_bins(rows)
    plot_population(p, rows, pop)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--pop", choices=["msp", "all"], default="msp")
    ap.add_argument("--nbins", type=int, default=12)
    ap.add_argument("--col", default=None)
    a = ap.parse_args()
    main(a.path, pop=a.pop, nbins=a.nbins, col=a.col)
