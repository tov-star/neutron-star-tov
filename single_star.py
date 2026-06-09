"""
single_star.py  --  build ONE neutron-star model for an EoS and plot its profiles.

Usage:
    python single_star.py <NAME> [rho_c]      # rho_c in g/cm^3, default 1e15
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

DEFAULT = "ABHT_QMCRMF1"


def main(name=DEFAULT, rho_c=1.0e15):
    eos = EOS.from_name(name)
    outdir = os.path.join("results", name)
    os.makedirs(outdir, exist_ok=True)

    M, R, (rs, Ps, ms) = solve_star(rho_c, eos, h=2.0e3, return_profile=True)
    print(f"{name}: rho_c={rho_c:.3e}  M={M / M_SUN:.3f} Msun  R={R / 1e5:.3f} km")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(rs / 1e5, Ps); ax1.set_yscale("log")
    ax1.set_xlabel("radius [km]"); ax1.set_ylabel("pressure [dyn/cm$^2$]")
    ax1.set_title("Pressure profile")
    ax2.plot(rs / 1e5, ms / M_SUN)
    ax2.set_xlabel("radius [km]"); ax2.set_ylabel("enclosed mass [M$_\\odot$]")
    ax2.set_title("Mass profile")
    fig.suptitle(f"{name}: M={M / M_SUN:.3f} M$_\\odot$, R={R / 1e5:.2f} km")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "single_star.png"), dpi=130)
    print(f"saved {outdir}/single_star.png")


if __name__ == "__main__":
    nm = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    rc = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0e15
    main(nm, rc)
