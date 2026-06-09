"""
build_eos.py  --  prepare a 1-D cold, beta-equilibrium EoS table for the solver.

Reads the raw CompOSE files in   eos_raw/<NAME>/   (eos.nb + eos.thermo) and
writes a standard table          eos_tables/<NAME>.dat   with columns:
    nb[fm^-3]   rho[g/cm^3]   P[dyn/cm^2]

This is the ONLY place that knows about CompOSE's raw format. Everything
downstream reads the standard .dat, so the solver is EoS-agnostic.

Auto-detects dimensionality from the number of rows in eos.thermo:
  * 1-D table  (rows == number of densities)        -> read P and e directly.
  * 3-D table  (T x nb x Yq, many rows per density)  -> reduce to cold beta-eq:
        cold slice = temperature index with minimum entropy
        beta-eq    = charge fraction minimising energy per baryon at each density
Then drops any non-monotonic crust points so P(rho) is strictly increasing.

CompOSE column convention (after the 3 index columns iT, inb, iYq):
    Q1 = p / n_b            -> column index 3
    Q2 = s / n_b (entropy)  -> column index 4
    Q7 = e/(n_b m_n) - 1    -> column index 9
"""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np

m_n = 1.674927498e-24
MeVfm3_to_dyn = 1.602176634e33
fm3_to_cm3 = 1.0e39


def _read_nb(path):
    raw = np.loadtxt(path)
    n = int(raw[1])
    return raw[2:2 + n]


def prepare(name, raw_dir="eos_raw", out_dir="eos_tables"):
    folder = os.path.join(raw_dir, name)
    nb = _read_nb(os.path.join(folder, "eos.nb"))
    npb = len(nb)

    cols = np.loadtxt(os.path.join(folder, "eos.thermo"),
                      skiprows=1, usecols=(0, 1, 2, 3, 4, 9))
    iT, inb, iYq = (cols[:, 0].astype(int),
                    cols[:, 1].astype(int),
                    cols[:, 2].astype(int))
    Q1, Q2, Q7 = cols[:, 3], cols[:, 4], cols[:, 5]
    assert inb.max() == npb, "thermo column 1 is not the density index (unexpected format)"

    if cols.shape[0] == npb:
        # ---------- 1-D table: already cold + beta-equilibrium ----------
        dim = "1-D"
        order = np.argsort(inb)
        P = Q1[order] * nb * MeVfm3_to_dyn
        rho = (Q7[order] + 1.0) * nb * fm3_to_cm3 * m_n
    else:
        # ---------- 3-D table: reduce to cold beta-equilibrium ----------
        dim = "3-D"
        ref = (inb == npb // 2) & (iYq == iYq.max() // 2)
        iT_cold = iT[ref][np.argmin(Q2[ref])]
        cold = (iT == iT_cold)
        inb_c, Q1_c, Q7_c = inb[cold], Q1[cold], Q7[cold]
        P = np.full(npb, np.nan)
        rho = np.full(npb, np.nan)
        for j in range(1, npb + 1):
            mm = (inb_c == j)
            if not np.any(mm):
                continue
            k = np.argmin(Q7_c[mm])              # beta-eq: min energy per baryon
            P[j - 1] = Q1_c[mm][k] * nb[j - 1] * MeVfm3_to_dyn
            rho[j - 1] = (Q7_c[mm][k] + 1.0) * nb[j - 1] * fm3_to_cm3 * m_n

    good = np.isfinite(P) & np.isfinite(rho) & (P > 0) & (rho > 0)
    nbk, rho, P = nb[good], rho[good], P[good]
    o = np.argsort(rho)
    nbk, rho, P = nbk[o], rho[o], P[o]

    # strictly-increasing filter (drops crust non-monotonic points)
    keep = [0]
    for i in range(1, len(P)):
        if P[i] > P[keep[-1]] and rho[i] > rho[keep[-1]]:
            keep.append(i)
    keep = np.array(keep)
    nbk, rho, P = nbk[keep], rho[keep], P[keep]
    assert np.all(np.diff(P) > 0) and np.all(np.diff(rho) > 0)

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{name}.dat")
    np.savetxt(out, np.column_stack([nbk, rho, P]),
               header=f"{name} ({dim}) cold beta-eq:  nb[fm^-3]  rho[g/cm^3]  P[dyn/cm^2]")
    print(f"[{name}] {dim} table, kept {len(P)}/{npb} points -> {out}")
    print(f"   rho {rho.min():.3e}..{rho.max():.3e} g/cm^3,"
          f"  P {P.min():.3e}..{P.max():.3e} dyn/cm^2")
    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        prepare(sys.argv[1])
    else:
        import catalog
        for row in catalog.load():
            try:
                prepare(row["name"])
            except (FileNotFoundError, OSError):
                print(f"[{row['name']}] no raw files in eos_raw/{row['name']}/ -- skipping")
