"""
compare_to_compose.py  --  overlay my M-R curve on the CompOSE reference (eos.mr).

Usage:
    python compare_to_compose.py            # ALL EoS in the catalog
    python compare_to_compose.py all        # same
    python compare_to_compose.py <NAME>     # just one EoS

For each EoS it reads  eos_tables/<NAME>.dat  and  eos_raw/<NAME>/eos.mr,
and writes  results/<NAME>/compare_mr.png.  EoS missing a prepared table or a
reference file are skipped with a note (so running "all" is always safe).
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


def main(name=DEFAULT):
    table = os.path.join("eos_tables", f"{name}.dat")
    ref_path = os.path.join("eos_raw", name, "eos.mr")
    if not os.path.exists(table):
        print(f"[{name}] no eos_tables/{name}.dat (run build_eos.py {name} first) -- skipping")
        return
    if not os.path.exists(ref_path):
        print(f"[{name}] no eos.mr reference in eos_raw/{name}/ -- skipping")
        return

    eos = EOS.from_name(name)
    outdir = os.path.join("results", name)
    os.makedirs(outdir, exist_ok=True)

    ref = np.loadtxt(ref_path)
    Rr, Mr = ref[:, 0], ref[:, 1]

    rho_c = np.logspace(np.log10(5e13), np.log10(eos.rho[-1] * 0.999), 55)
    M, R = [], []
    for rc in rho_c:
        m, r = solve_star(rc, eos, h=5.0e3)
        M.append(m / M_SUN); R.append(r / 1e5)
    M, R = np.array(M), np.array(R)

    print(f"[{name}] reference max mass = {Mr.max():.3f} Msun, "
          f"mine = {np.nanmax(M):.3f} Msun")

    plt.figure(figsize=(7.5, 6))
    plt.plot(Rr, Mr, "s", ms=5, mfc="none", color="crimson",
             label="CompOSE eos.mr (reference)")
    plt.plot(R, M, "-", lw=1.6, color="steelblue", label="my solver")
    plt.xlabel("radius [km]"); plt.ylabel("mass [M$_\\odot$]")
    plt.title(f"My M-R vs CompOSE reference: {name}")
    plt.xlim(9, 20); plt.ylim(0, max(2.2, np.nanmax(M) * 1.1))
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "compare_mr.png"), dpi=130)
    print(f"[{name}] saved {outdir}/compare_mr.png")


def run_all():
    import catalog
    for row in catalog.load():
        main(row["name"])


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    if arg == "all":
        run_all()
    else:
        main(arg)
