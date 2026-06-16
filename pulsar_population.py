"""
pulsar_population.py  --  Part 2: the neutron-star (pulsar) rotation population.

Goal: from the ATNF pulsar catalogue, show the period histogram (it is strongly
bimodal) and extract the mean rotation period -- and, crucially, the mean period
of the MILLISECOND population, which is the one relevant to the Kusenko spin-up
mechanism (only MSPs sit close enough to mass shedding to be spun up to ejection).

------------------------------------------------------------------------------
HOW TO GET THE DATA  (do this once, on your machine, with a network connection)
  Web interface:  https://www.atnf.csiro.au/research/pulsar/psrcat/
    - In "Output Parameters" tick  P0  (barycentric period, seconds).
    - Optionally tick  NAME  and  BINARY.
    - Set output format to a plain table or CSV and save it, e.g.  psrcat_P0.csv
  Command line (if you installed psrcat):
    psrcat -c "NAME P0" -o short  >  psrcat_P0.txt

Then:
    python pulsar_population.py psrcat_P0.csv
    python pulsar_population.py psrcat_P0.txt        # whitespace table also works

The parser is permissive: it reads every numeric token that looks like a period
in seconds (1e-3 s ... 100 s) from the file, whatever the exact column layout.
If you know the column name (CSV header), pass it:  --col P0
------------------------------------------------------------------------------

For reference (ATNF, ~2025): ~3000+ pulsars, ~560 MSPs. Normal pulsars peak near
P ~ 0.5 s (log-normal); MSPs (P < 30 ms) peak near P ~ 3-5 ms; fastest ~1.4 ms.
Fuller-Kusenko-Takhistov use a fiducial P0 ~ 1 ms for the ejection estimate.
"""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import sys
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MSP_CUTOFF = 30e-3     # s  (P < 30 ms == "millisecond pulsar"; excludes the Crab)


def load_periods(path, col=None):
    """Return a 1-D array of spin periods [s]. Tries pandas (CSV with header),
    then falls back to grabbing every float in the plausible period range."""
    # 1) try a real CSV with a named/обvious period column
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
            p = p[np.isfinite(p)]
            p = p[(p > 1e-3) & (p < 100)]
            if p.size:
                return p
    except Exception:
        pass

    # 2) permissive fallback: every float in [1e-3, 100] s on each line
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
                if 1e-3 < v < 100:        # plausible pulsar period in seconds
                    periods.append(v)
                    break                 # one period per line (the P0 column)
    return np.array(periods, dtype=float)


def summarize(p):
    msp = p[p < MSP_CUTOFF]
    normal = p[p >= MSP_CUTOFF]
    print(f"total pulsars read           : {p.size}")
    print(f"  millisecond (P<30 ms)      : {msp.size}")
    print(f"  normal      (P>=30 ms)     : {normal.size}")
    print(f"mean   period (all)          : {np.mean(p)*1e3:.2f} ms")
    print(f"median period (all)          : {np.median(p)*1e3:.2f} ms")
    if msp.size:
        print(f"mean   period (MSPs)         : {np.mean(msp)*1e3:.3f} ms   <-- use this as P0")
        print(f"median period (MSPs)         : {np.median(msp)*1e3:.3f} ms")
        print(f"fastest pulsar               : {np.min(p)*1e3:.3f} ms")
        omega = 2*np.pi/np.mean(msp)
        print(f"Omega0 (mean MSP)            : {omega:.3e} rad/s")
    return msp, normal


def plot_hist(p, outpath="results/pulsar_period_hist.png"):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    logp = np.log10(p)               # log10(P / s)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(logp, bins=60, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(np.log10(MSP_CUTOFF), color="crimson", ls="--",
               label=f"MSP cut (30 ms)")
    ax.axvline(np.log10(np.median(p)), color="k", ls=":",
               label=f"median = {np.median(p)*1e3:.0f} ms")
    ax.set_xlabel(r"$\log_{10}(P\,/\,\mathrm{s})$")
    ax.set_ylabel("number of pulsars")
    ax.set_title("ATNF pulsar spin-period distribution")
    # secondary tick labels in physical units
    ticks = np.array([1e-3, 1e-2, 1e-1, 1e0, 1e1])
    ax.set_xticks(np.log10(ticks))
    ax.set_xticklabels(["1 ms", "10 ms", "0.1 s", "1 s", "10 s"])
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=130)
    plt.close(fig)
    print(f"saved {outpath}")


def main(path, col=None):
    p = load_periods(path, col=col)
    if p.size == 0:
        print("No periods parsed. Check the file, or pass --col <name> for a CSV.")
        return
    summarize(p)
    plot_hist(p)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    col = None
    if "--col" in sys.argv:
        col = sys.argv[sys.argv.index("--col") + 1]
    if not args:
        print(__doc__)
        print("usage: python pulsar_population.py <psrcat_export> [--col P0]")
    else:
        main(args[0], col=col)
