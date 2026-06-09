"""
single_star.py  --  build ONE neutron-star model and plot its profiles.

Run:
    python single_star.py

If the CompOSE files `eos.nb` and `eos.thermo` are present, the real APR
EOS is used. Otherwise it falls back to the toy polytrope so you can still
test the machinery.  Cross-check: the printed (M, R) should land on one
point of the published APR mass-radius curve.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")               # headless: save figures instead of showing
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
    print("CompOSE files not found -- falling back to toy polytrope EOS.")
    return EOS.polytrope()


def main():
    eos = load_eos()
    eos.summary()

    rho_c = 1.0e15                              # g/cm^3
    rho_stop = 1.0e9 if eos.mode == "polytrope" else None

    M, R, (rs, Ps, ms) = solve_star(
        rho_c, eos, h=2.0e3, rho_stop=rho_stop, return_profile=True)

    print("\nNeutron Star Solution")
    print("---------------------")
    print(f"central density = {rho_c:.3e} g/cm^3")
    print(f"radius          = {R/1e5:.3f} km")
    print(f"mass            = {M/M_SUN:.3f} Msun")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(rs / 1e5, Ps)
    ax1.set_xlabel("radius [km]"); ax1.set_ylabel("pressure [dyn/cm$^2$]")
    ax1.set_yscale("log"); ax1.set_title("Pressure profile")
    ax2.plot(rs / 1e5, ms / M_SUN)
    ax2.set_xlabel("radius [km]"); ax2.set_ylabel("enclosed mass [M$_\\odot$]")
    ax2.set_title("Mass profile")
    fig.suptitle(f"Single star ({eos.mode} EOS): M={M/M_SUN:.3f} M$_\\odot$, "
                 f"R={R/1e5:.2f} km")
    fig.tight_layout()
    fig.savefig("single_star.png", dpi=130)
    print("\nsaved single_star.png")


if __name__ == "__main__":
    main()
