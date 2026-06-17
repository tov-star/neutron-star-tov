"""
ejection_conditions.py  --  read a finished survey grid and report the EJECTION
CONDITIONS for an EoS: the critical M_BH at which v_eq/v_K first reaches 1, as a
function of neutron-star mass, initial spin period P0, and spin-up model. The
ejection ratio scales as 1/P0, so a single survey grid gives every period for
free (no re-running).

Usage
  python ejection_conditions.py results/APR/accretion_grid.npz
  python ejection_conditions.py results/APR/accretion_grid.npz \
         --masses 1.2 1.4 1.6 1.8 2.0 --periods 1.0 1.5 2.0 3.0 6.4
  python ejection_conditions.py results/APR/accretion_grid.npz \
         --bins results/period_bins.csv          # population-weighted condition

Reads P0_ref (the period the grid was computed at) from meta.npz next to the grid.
"""
import os, argparse
import numpy as np


def critical_mbh(mbh, ratio_row, scale):
    """First M_BH where ratio_row*scale >= 1 (linear interp). NaN if never."""
    r = ratio_row * scale
    hit = np.where(r >= 1.0)[0]
    if hit.size == 0:
        return np.nan
    i = hit[0]
    if i == 0:
        return mbh[0]
    f = (1.0 - r[i - 1]) / (r[i] - r[i - 1])
    return mbh[i - 1] + f * (mbh[i] - mbh[i - 1])


def load_grid(grid_path):
    d = np.load(grid_path)
    meta_path = os.path.join(os.path.dirname(grid_path), "meta.npz")
    P0_ref = float(np.load(meta_path)["P0"]) if os.path.exists(meta_path) else 6.4e-3
    return d, P0_ref


def table(d, P0_ref, masses, periods_ms):
    mbh, M = d["mbh"], d["M"]
    for key, name in (("ratio_sj", "specific_j (conservative)"),
                      ("ratio_rg", "rigid (efficient transfer)")):
        print(f"\nCritical M_BH [Msun] for ejection -- {name}")
        print(f"  (P0_ref = {P0_ref*1e3:.2f} ms;  '-- none --' = no ejection for "
              f"M_BH <= {mbh[-1]:.2f})")
        hdr = "  M[Msun] | " + " ".join(f"{p:>5.1f}ms" for p in periods_ms)
        print(hdr); print("  " + "-" * (len(hdr) - 2))
        for tm in masses:
            j = int(np.nanargmin(np.abs(M[:, 0] - tm)))
            row = d[key][j, :]
            cells = []
            for P in periods_ms:
                mc = critical_mbh(mbh, row, P0_ref / (P * 1e-3))
                cells.append(f"{mc:7.3f}" if np.isfinite(mc) else "  none ")
            print(f"  {M[j,0]:7.2f} | " + " ".join(cells))


def population_weighted(d, P0_ref, bins_path, target_M):
    rows = np.genfromtxt(bins_path, delimiter=",", names=True)
    P_rep = np.atleast_1d(rows["P_rep_ms"]) * 1e-3
    weight = np.atleast_1d(rows["weight"])
    mbh, M = d["mbh"], d["M"]
    j = int(np.nanargmin(np.abs(M[:, 0] - target_M)))
    print(f"\nPopulation-weighted ejection fraction at M = {M[j,0]:.2f} Msun")
    print(f"  (fraction of the binned pulsar population that sheds matter)")
    for key, name in (("ratio_sj", "specific_j"), ("ratio_rg", "rigid")):
        row = d[key][j, :]
        print(f"  {name}:")
        print("    M_BH :  " + " ".join(f"{mb:5.2f}" for mb in mbh))
        fr = []
        for i in range(mbh.size):
            ejects = (row[i] * (P0_ref / P_rep) >= 1.0)        # per bin at this M_BH
            fr.append(np.nansum(weight[ejects]))
        print("    frac :  " + " ".join(f"{x:5.2f}" for x in fr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grid")
    ap.add_argument("--masses", type=float, nargs="+",
                    default=[1.2, 1.4, 1.6, 1.8, 2.0])
    ap.add_argument("--periods", type=float, nargs="+",
                    default=[1.0, 1.5, 2.0, 3.0, 6.4], help="ms")
    ap.add_argument("--bins", default=None, help="period_bins.csv for weighting")
    ap.add_argument("--target-M", type=float, default=1.4)
    a = ap.parse_args()
    d, P0_ref = load_grid(a.grid)
    table(d, P0_ref, a.masses, a.periods)
    if a.bins:
        population_weighted(d, P0_ref, a.bins, a.target_M)


if __name__ == "__main__":
    main()
