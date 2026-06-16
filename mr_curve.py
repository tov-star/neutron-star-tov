"""
mr_curve.py  --  mass-radius relation for one EoS.

Usage:
    python mr_curve.py <NAME>          # e.g. ABHT_QMCRMF1 or APR

Reads   eos_tables/<NAME>.dat
Writes  results/<NAME>/mr_curve.png  and  results/<NAME>/mr_points.dat
Also checks the maximum mass against the catalog reference and the heaviest
observed pulsar (the project's EoS-validity test).
"""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eos import EOS
from solver import solve_star, M_SUN
import catalog

DEFAULT = "APR"


def main(name=DEFAULT):
    eos = EOS.from_name(name)
    outdir = os.path.join("results", name)
    os.makedirs(outdir, exist_ok=True)

    rho_c = np.logspace(np.log10(3.4e14), np.log10(eos.rho[-1] * 0.99), 60)
    M, R = [], []
    print(f"{'rho_c [g/cm^3]':>16}{'M [Msun]':>12}{'R [km]':>10}")
    for rc in rho_c:
        m, r = solve_star(rc, eos, h=2.0e2)
        M.append(m / M_SUN); R.append(r / 1e5)
        print(f"{rc:>16.4e}{m / M_SUN:>12.4f}{r / 1e5:>10.3f}")
    M, R = np.array(M), np.array(R)

    imax = np.nanargmax(M)
    print("\n=== MAX mass point ===")
    print(f"rho_c = {rho_c[imax]:.4e} g/cm^3")
    print(f"M     = {M[imax]:.4f} Msun")
    print(f"R     = {R[imax]:.3f} km")

    try:
        meta = catalog.get(name)
        print(f"reference Mmax (CompOSE) = {meta['ref_Mmax']} Msun")
        passes = M[imax] >= catalog.OBS_MAX_MASS
        print(f"clears heaviest pulsar ({catalog.OBS_MAX_MASS} Msun)? "
              f"{'YES' if passes else 'NO'}")
    except KeyError:
        pass

    np.savetxt(os.path.join(outdir, "mr_points.dat"),
               np.column_stack([rho_c, M, R]),
               header="rho_c[g/cm^3]  M[Msun]  R[km]")

    plt.figure(figsize=(7.5, 6))
    plt.plot(R, M, "-o", ms=3, label=name)
    plt.scatter([R[imax]], [M[imax]], color="crimson", zorder=5,
                label=f"max mass = {M[imax]:.2f} M$_\\odot$")
    plt.xlabel("radius [km]"); plt.ylabel("mass [M$_\\odot$]")
    plt.title(f"Mass-radius relation: {name}")
    plt.xlim(9, 16); plt.ylim(0, max(2.5, M[imax] * 1.1))
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "mr_curve.png"), dpi=130)
    print(f"saved {outdir}/mr_curve.png and mr_points.dat")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT)
