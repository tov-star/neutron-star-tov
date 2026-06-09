"""
mr_curve.py  --  sweep central density to trace the mass-radius relation.

Run:
    python mr_curve.py

Cross-checks (project Part 2):
  * The curve should overlap the published APR M-R relation when using the
    real CompOSE EOS.
  * The peak of the curve is the maximum (TOV) mass -- compare it against the
    observed "most massive neutron star" to decide if the EOS is allowed.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eos import EOS
from solver import solve_star, M_SUN


def load_eos():
    if os.path.exists("eos_cold_beta.dat"):
        print("Using reduced cold beta-equilibrium APR table.")
        return EOS.from_table()
    if os.path.exists("eos.nb") and os.path.exists("eos.thermo"):
        print("Reducing 3D CompOSE table -> cold beta-eq (run build_eos.py).")
        import build_eos; build_eos.build()
        return EOS.from_table()
    print("CompOSE files not found -- using toy polytrope EOS.")
    return EOS.polytrope()


def sweep(eos, relativistic, rho_c_values, rho_stop):
    M, R = [], []
    for rc in rho_c_values:
        m, r = solve_star(rc, eos, h=2.0e3, rho_stop=rho_stop,
                          relativistic=relativistic)
        M.append(m / M_SUN); R.append(r / 1e5)
    return np.array(R), np.array(M)


def main():
    eos = load_eos()
    rho_stop = 1.0e9 if eos.mode == "polytrope" else None

    lo = 3.4e14 if eos.mode != "polytrope" else 2.0e14
    rho_c_values = np.logspace(np.log10(lo), np.log10(eos.rho[-1] * 0.99), 60)

    R_gr, M_gr = sweep(eos, True, rho_c_values, rho_stop)

    imax = np.nanargmax(M_gr)
    print(f"Maximum (TOV) mass = {M_gr[imax]:.3f} Msun "
          f"at R = {R_gr[imax]:.2f} km, rho_c = {rho_c_values[imax]:.3e}")

    name = "APR" if eos.mode == "compose" else "polytrope"
    plt.figure(figsize=(7.5, 6))
    plt.plot(R_gr, M_gr, "-o", ms=3, label=f"{name} EOS")
    plt.scatter([R_gr[imax]], [M_gr[imax]], color="crimson", zorder=5,
                label=f"max mass = {M_gr[imax]:.2f} M$_\\odot$")
    plt.xlabel("radius [km]"); plt.ylabel("mass [M$_\\odot$]")
    plt.title(f"Mass-radius relation ({name} EOS)")
    plt.xlim(9, 16); plt.ylim(0, max(2.5, M_gr[imax] * 1.1))
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("mr_curve.png", dpi=130)
    print("saved mr_curve.png")


if __name__ == "__main__":
    main()
