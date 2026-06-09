"""
build_eos.py  --  reduce the CompOSE 3D APR table to a 1D cold,
beta-equilibrium neutron-star EOS that the solver can use.

The general-purpose CompOSE table `eos.thermo` is 3-D:
    (temperature) x (baryon density) x (charge fraction).
For a neutron star we want a single 1-D curve P(rho):

  1. COLD slice: take the lowest-temperature index (iT = 1, identified as
     the slice with minimum entropy per baryon).
  2. BETA-EQUILIBRIUM composition: at each density, pick the charge fraction
     that MINIMISES the energy per baryon. At T = 0 that minimum is exactly
     the beta-equilibrium state (valid here because the table's energy
     includes the electrons -- confirmed by the selected matter being
     strongly neutron-rich).
  3. MONOTONICITY: drop the few crust-region points where P is not strictly
     increasing (a known artifact of the simple per-density selection across
     the crust); the core, which sets M and R, is unaffected.

Output: eos_cold_beta.dat  with columns  nb[fm^-3]  rho[g/cm^3]  P[dyn/cm^2].

NOTE (physics to confirm with mentor): which temperature index counts as
"cold", and the crust treatment, are modelling choices. The dedicated cold
beta-equilibrium table from CompOSE's "thermodynamic conditions" tool is an
independent way to get the same curve.
"""

import numpy as np

m_n = 1.674927498e-24
MeVfm3_to_dyn = 1.602176634e33
fm3_to_cm3 = 1.0e39


def build(nb_file="eos.nb", thermo_file="eos.thermo",
          out="eos_cold_beta.dat"):
    nb_raw = np.loadtxt(nb_file)
    npb = int(nb_raw[1])
    nb = nb_raw[2:2 + npb]

    cols = np.loadtxt(thermo_file, skiprows=1, usecols=(0, 1, 2, 3, 4, 9))
    iT, inb, iYq = cols[:, 0].astype(int), cols[:, 1].astype(int), cols[:, 2].astype(int)
    Q1, Q2, Q7 = cols[:, 3], cols[:, 4], cols[:, 5]

    # 1. cold slice = minimum entropy (checked at a reference grid point)
    ref = (inb == npb // 2) & (iYq == 30)
    iT_cold = iT[ref][np.argmin(Q2[ref])]

    cold = (iT == iT_cold)
    inb_c, iYq_c, Q1_c, Q7_c = inb[cold], iYq[cold], Q1[cold], Q7[cold]

    # 2. beta-equilibrium = minimum energy per baryon over charge fraction
    P = np.full(npb, np.nan)
    rho = np.full(npb, np.nan)
    for j in range(1, npb + 1):
        mm = (inb_c == j)
        if not np.any(mm):
            continue
        k = np.argmin(Q7_c[mm])
        P[j - 1] = Q1_c[mm][k] * nb[j - 1] * MeVfm3_to_dyn
        rho[j - 1] = (Q7_c[mm][k] + 1.0) * nb[j - 1] * fm3_to_cm3 * m_n

    good = np.isfinite(P) & np.isfinite(rho) & (P > 0) & (rho > 0)
    nb, rho, P = nb[good], rho[good], P[good]
    order = np.argsort(rho)
    nb, rho, P = nb[order], rho[order], P[order]

    # 3. greedy strictly-increasing filter (drops the crust non-monotonic pts)
    keep = [0]
    for i in range(1, len(P)):
        if P[i] > P[keep[-1]] and rho[i] > rho[keep[-1]]:
            keep.append(i)
    keep = np.array(keep)
    nb, rho, P = nb[keep], rho[keep], P[keep]

    assert np.all(np.diff(P) > 0) and np.all(np.diff(rho) > 0)
    np.savetxt(out, np.column_stack([nb, rho, P]),
               header="nb[fm^-3]  rho[g/cm^3]  P[dyn/cm^2]  (cold beta-eq APR)")
    print(f"cold slice iT={iT_cold}; kept {len(P)} of {npb} points")
    print(f"rho: {rho.min():.3e} .. {rho.max():.3e} g/cm^3")
    print(f"P  : {P.min():.3e} .. {P.max():.3e} dyn/cm^2")
    print(f"saved {out}")


if __name__ == "__main__":
    build()
