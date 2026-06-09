"""
convergence.py  --  does the solver converge at the expected order?

Project cross-check #4: "If you halve the step size, the error should drop by
2^4 = 16 for a 4th-order scheme."

What you will see, and why it matters:

  * The INTERIOR RK4 integration is genuinely 4th order (ratio ~16) -- verified
    below on a segment away from the center.

  * The FULL center-to-surface mass is only ~2nd order (ratio ~4). This is NOT
    a bug in RK4. It is limited by two 2nd-order pieces at the boundaries:
        (a) the r=0 start: seeding m0 = (4/3) pi r0^3 rho_c is only the
            leading term of the central Taylor series;
        (b) the surface: locating it by LINEAR interpolation of the last step.
    A scheme is only as high-order as its lowest-order component.

  TO RECOVER 4th ORDER GLOBALLY (next improvement):
    - take the first step off-center from a Taylor-series expansion of P(r),
      m(r) about r=0, and
    - find the surface with a higher-order (e.g. cubic-Hermite) crossing.
"""

import numpy as np
from eos import EOS
from tov import tov_rhs
from rk4 import rk4_step
from solver import solve_star, M_SUN


def full_star_convergence(eos, rho_c=1e15, rho_stop=1e9):
    print("FULL star (center -> surface): convergence of total mass")
    Ms = []
    for h in [8e3, 4e3, 2e3, 1e3]:
        M, R = solve_star(rho_c, eos, h=h, rho_stop=rho_stop)
        Ms.append(M / M_SUN)
        print(f"  h={h:7.0f} cm   M={M/M_SUN:.7f} Msun   R={R/1e5:.4f} km")
    d = np.abs(np.diff(Ms))
    for i in range(len(d) - 1):
        print(f"  successive-difference ratio: {d[i]/d[i+1]:.1f}  "
              f"(2nd order ~4, 4th order ~16)")


def interior_convergence(eos, rho_c=1e15):
    print("\nINTERIOR segment (2 km -> 4 km, away from center): pure RK4 order")

    def run(r0, P0, m0, r_end, N):
        h = (r_end - r0) / N
        r, P, m = r0, P0, m0
        for _ in range(N):
            P, m = rk4_step(r, P, m, h, eos, tov_rhs)
            r += h
        return P

    # accurate state at exactly 2 km
    r0 = 1.0
    P0, m0 = eos.pressure(rho_c), (4 * np.pi / 3) * rho_c * r0**3
    h = (2.0e5 - r0) / 400000
    r, P, m = r0, P0, m0
    for _ in range(400000):
        P, m = rk4_step(r, P, m, h, eos, tov_rhs); r += h
    P2, m2 = P, m

    ref = run(2.0e5, P2, m2, 4.0e5, 40000)
    prev = None
    for N in [125, 250, 500, 1000]:
        err = abs(run(2.0e5, P2, m2, 4.0e5, N) - ref)
        ratio = (prev / err) if prev else float("nan")
        print(f"  N={N:5d}  |err P|={err:.3e}  ratio={ratio:5.1f}")
        prev = err


if __name__ == "__main__":
    eos = EOS.polytrope()       # data-free machinery test
    full_star_convergence(eos)
    interior_convergence(eos)
