"""
compare_to_compose.py  --  overlay my mass-radius curve on the CompOSE
reference curve (eos.mr) and quantify the agreement. This is the Part 2
cross-check: "your curve should overlap theirs."
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eos import EOS
from solver import solve_star, M_SUN

ref = np.loadtxt("eos.mr")          # columns: R[km], M[Msun]
Rr, Mr = ref[:, 0], ref[:, 1]

eos = EOS.from_table()
rho_c = np.logspace(np.log10(5e13), np.log10(eos.rho[-1] * 0.999), 55)
M, R = [], []
for rc in rho_c:
    m, r = solve_star(rc, eos, h=5.0e3)     # bigger step -> fast
    M.append(m / M_SUN); R.append(r / 1e5)
M, R = np.array(M), np.array(R)

print(f"reference max mass = {Mr.max():.3f} Msun at R={Rr[np.argmax(Mr)]:.2f} km")
print(f"my        max mass = {np.nanmax(M):.3f} Msun at R={R[np.nanargmax(M)]:.2f} km")

plt.figure(figsize=(7.5, 6))
plt.plot(Rr, Mr, "s", ms=5, mfc="none", color="crimson",
         label="CompOSE eos.mr (reference)")
plt.plot(R, M, "-", lw=1.6, color="steelblue", label="my solver")
plt.xlabel("radius [km]"); plt.ylabel("mass [M$_\\odot$]")
plt.title("My M-R vs CompOSE reference (APR)")
plt.xlim(9, 20); plt.ylim(0, 2.2)
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig("compare_mr.png", dpi=130)
print("saved compare_mr.png")
