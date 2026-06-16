"""
solver.py  --  integrate one stellar model from the center to the surface.

Boundary conditions at the center (guiding question on BCs):
    rho(0) = rho_c,   m(0) = 0,   P(0) = P(rho_c)

The TOV right-hand side has r^2 and m in denominators, so r = 0 is a
removable singularity. We sidestep it by starting at a tiny radius r0 and
seeding the enclosed mass with the uniform-sphere value
    m0 = (4/3) pi r0^3 rho_c
which is exact in the r -> 0 limit.

The surface is where the pressure drops to P_surface (the lowest tabulated
pressure, or a chosen rho_stop). We linearly interpolate the last step so
the radius/mass aren't quantized by the step size h.
"""

import numpy as np
from rk4 import rk4_step
from tov import tov_rhs, newtonian_rhs, G, c

M_SUN = 1.98847e33     # g


def solve_star(rho_c, eos, M_BH=0.0, h=2.0e3, rho_stop=None,
               relativistic=True, r0=1.0, max_steps=500000,
               return_profile=False):
    """
    Integrate one star of central density rho_c.

    If M_BH > 0, a primordial black hole of that mass (in grams) sits at the
    center: the integration starts just outside its Schwarzschild radius with
    the enclosed mass seeded to M_BH. With M_BH = 0 this is the ordinary star.

    Returns (M_star [g], R_star [cm]); optionally the (r, P, m) profile.
    M_star is the TOTAL gravitational mass (black hole + fluid).
    """
    rhs = tov_rhs if relativistic else newtonian_rhs

    # Surface threshold in pressure
    if rho_stop is None:
        P_stop = eos.P_surface
    else:
        P_stop = eos.pressure(rho_stop)

    # --- inner boundary condition ---
    # Ordinary star (M_BH = 0): start at a tiny r0 with the uniform-sphere
    # seed m0 = (4/3) pi r0^3 rho_c, and integrate with a fixed step.
    #
    # With a central black hole: the fluid only exists OUTSIDE the event
    # horizon r_s = 2 G M_BH / c^2, so we start just outside it and seed the
    # enclosed mass with the hole alone (m = M_BH) -- there is no fluid to
    # lump inside the horizon. rho_c is then the fluid density at this inner
    # edge. Because the pressure gradient is steep right outside the horizon,
    # we take small steps there (h ~ r) and let them grow to the full h
    # farther out.
    P = eos.pressure(rho_c)
    if M_BH > 0.0:
        r_s = 2.0 * G * M_BH / c**2
        r0 = 2.0 * r_s
        m = M_BH
        adaptive = True
    else:
        m = (4.0 * np.pi / 3.0) * rho_c * r0**3
        adaptive = False
    r = r0

    rs, Ps, ms = [r], [P], [m]

    for _ in range(max_steps):
        # near the horizon use a small step (~r), growing to h farther out
        h_step = min(h, 0.02 * r) if adaptive else h

        P_new, m_new = rk4_step(r, P, m, h_step, eos, rhs)

        # crossed the surface -> interpolate to P_stop and finish
        if P_new <= P_stop:
            frac = (P - P_stop) / (P - P_new)
            R_star = r + frac * h_step
            M_star = m + frac * (m_new - m)
            if return_profile:
                rs.append(R_star); Ps.append(P_stop); ms.append(M_star)
                return M_star, R_star, (np.array(rs), np.array(Ps), np.array(ms))
            return M_star, R_star

        r += h_step
        P, m = P_new, m_new
        rs.append(r); Ps.append(P); ms.append(m)

    # did not converge within max_steps
    if return_profile:
        return np.nan, np.nan, (np.array(rs), np.array(Ps), np.array(ms))
    return np.nan, np.nan
