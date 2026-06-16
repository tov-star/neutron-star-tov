"""
pbh_mr.py  --  mass-radius curves for a neutron star with a central
primordial black hole, for several black-hole masses.

Usage:
    python pbh_mr.py <NAME> [M_BH1 M_BH2 ...]      # M_BH values in solar masses

For each M_BH it sweeps the inner-edge fluid density rho_c, integrates the
star (black hole + surrounding fluid), and plots total gravitational mass vs
radius. With M_BH = 0 this is the ordinary mass-radius curve.

Writes results/<NAME>/pbh_mr.png
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

DEFAULT = "APR"


def main(name=DEFAULT, mbh_list=(0, 0.1, 0.2, 0.3)):
    eos = EOS.from_name(name)
    outdir = os.path.join("results", name)
    os.makedirs(outdir, exist_ok=True)

    rho_c = np.logspace(np.log10(3.4e14), np.log10(eos.rho[-1] * 0.99), 50)

    plt.figure(figsize=(7.5, 6))
    print(f"{'M_BH[Msun]':>11}{'max M_tot':>11}{'R[km]':>9}")
    for mbh in mbh_list:
        M_BH = mbh * M_SUN
        M, R = [], []
        for rc in rho_c:
            m, r = solve_star(rc, eos, M_BH=M_BH, h=2.0e3)
            M.append(m / M_SUN); R.append(r / 1e5)
        M, R = np.array(M), np.array(R)
        i = np.nanargmax(M)
        print(f"{mbh:>11.3g}{M[i]:>11.4f}{R[i]:>9.3f}")
        plt.plot(R, M, "-", lw=1.6, label=f"M_BH = {mbh:g} M$_\\odot$")

    plt.xlabel("radius [km]"); plt.ylabel("total mass [M$_\\odot$]")
    plt.title(f"Neutron star with a central PBH: {name}")
    plt.xlim(6, 16); plt.ylim(0, 2.2)
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "pbh_mr.png"), dpi=130)
    print(f"saved {outdir}/pbh_mr.png")


if __name__ == "__main__":
    nm = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    mbhs = [float(x) for x in sys.argv[2:]] if len(sys.argv) > 2 else (0.0, 0.1, 0.2, 0.3, 1)
    main(nm, mbhs)
