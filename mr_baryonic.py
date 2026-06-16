"""
mr_baryonic.py  --  mass-radius diagram coloured by BARYONIC mass.

For each central black-hole mass it sweeps the central density, builds the star,
and records three numbers per star:
    R          radius                       -> x axis
    M_total    total gravitational mass     -> y axis
    M_baryonic rest mass of the fluid baryons -> colour

The gap between M_total and the colour at a point is that star's gravitational
binding energy. A line of constant colour (constant baryon number) is an
accretion track: how a star of fixed baryon content moves as the hole grows.

Usage:
    python mr_baryonic.py <NAME> [M_BH1 M_BH2 ...]    # M_BH in solar masses
Writes results/<NAME>/mr_baryonic.png
"""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eos import EOS
from solver import solve_star, baryonic_mass, M_SUN

DEFAULT = "APR"


def main(name=DEFAULT, mbh_list=(0.0, 0.1, 0.2, 0.3)):
    eos = EOS.from_name(name)
    outdir = os.path.join("results", name)
    os.makedirs(outdir, exist_ok=True)

    rho_c = np.logspace(np.log10(3.4e14), np.log10(eos.rho[-1] * 0.99), 45)

    allR, allMt, allMb = [], [], []
    fig, ax = plt.subplots(figsize=(8, 6))
    for mbh in mbh_list:
        M_BH = mbh * M_SUN
        R, Mt, Mb = [], [], []
        for rc in rho_c:
            m, r, prof = solve_star(rc, eos, M_BH=M_BH, h=2.0e3,
                                    return_profile=True)
            R.append(r / 1e5)
            Mt.append(m / M_SUN)
            Mb.append(baryonic_mass(eos, *prof) / M_SUN)
        ax.plot(R, Mt, "-", color="0.7", lw=0.8, zorder=1)   # faint guide line
        allR += R; allMt += Mt; allMb += Mb

    sc = ax.scatter(allR, allMt, c=allMb, cmap="viridis", s=22, zorder=2)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("baryonic mass  M$_b$  [M$_\\odot$]")
    ax.set_xlabel("radius  R  [km]")
    ax.set_ylabel("total gravitational mass  M$_{tot}$  [M$_\\odot$]")
    ax.set_title(f"M-R coloured by baryonic mass: {name}\n"
                 f"(curves = M_BH = {', '.join(str(m) for m in mbh_list)} M_sun)")
    ax.set_xlim(6, 16); ax.set_ylim(0, 2.4); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "mr_baryonic.png"), dpi=130)
    print(f"saved {outdir}/mr_baryonic.png")


if __name__ == "__main__":
    nm = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    mbhs = [float(x) for x in sys.argv[2:]] if len(sys.argv) > 2 else (0.0, 0.1, 0.2, 0.3)
    main(nm, mbhs)
