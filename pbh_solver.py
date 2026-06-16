"""
pbh_solver.py  --  TOV neutron star with a central black hole (PBH extension).

This finishes the Part-2 PBH step. It plugs straight into the existing engine:
it reuses tov_rhs / rk4_step and falls back to solver.solve_star when M_BH = 0,
so the no-BH limit is guaranteed to match the base solver bit-for-bit.

Guiding questions this touches:
  #8  role of the black hole          -> it is the inner boundary of the matter
  #9  what happens to the BH mass     -> it is the seed for m(r) at the inner edge
  #10 how R_BH scales with M_BH       -> R_BH = 2 G M_BH / c^2  (Schwarzschild)
  #12 baryonic vs gravitational mass  -> m(r) here is the *gravitational* (ADM)
                                         mass; the enclosed-mass seed is M_BH.

----------------------------------------------------------------------------
IMPORTANT PHYSICS NOTE (read before trusting the output of solve_star_with_pbh):

The sheet's literal inner boundary condition is
    rho(R_BH) = rho_c,  m(R_BH) = M_BH,   R_BH = 2 G M_BH / c^2.
That places nuclear-density matter *at the hole's own horizon*. There the TOV
redshift factor (1 - 2Gm/rc^2) -> 0, so dP/dr -> -inf. The pressure then falls
LOGARITHMICALLY and crosses zero after r grows by only ~10%, independent of the
step size. Result: the matter layer is negligibly thin and M_total -> M_BH.
You cannot statically support an extended high-density atmosphere against a
horizon. This is intrinsic, not a discretisation error (verified: insensitive
to eps and h).

The physically intended model for the ejection study is almost certainly the
CONSUMPTION-RADIUS construction instead: the BH sits at the radius r_b where the
ordinary star's enclosed mass equals M_BH (r_b >> R_BH for a sub-solar BH, so
the redshift factor is fine), and the envelope r_b < r < R_surf is kept. See
solve_star_consumed() below; that is the one that gives a smooth M_BH response
consistent with Fuller-Kusenko Eqs. (11-13).
----------------------------------------------------------------------------
"""

import numpy as np
from rk4 import rk4_step
from tov import tov_rhs, newtonian_rhs
from solver import solve_star, M_SUN

G = 6.67430e-8             # cm^3 g^-1 s^-2
c = 2.99792458e10          # cm/s


def schwarzschild_radius(M_BH):
    """R_BH = 2 G M_BH / c^2   (cm).  Guiding question #10."""
    return 2.0 * G * M_BH / c**2


# ---------------------------------------------------------------------------
# Route A: literal horizon boundary condition (sheet, verbatim).
#          Correct code; demonstrates the horizon-support pathology above.
# ---------------------------------------------------------------------------
def solve_star_with_pbh(M_BH, rho_c, eos, h=2.0e3, rho_stop=None,
                        relativistic=True, eps=1.0e-3, max_steps=2_000_000,
                        h_ramp=True, return_profile=False):
    """
    Integrate outward from r0 = R_BH*(1+eps) with m(r0)=M_BH, P(r0)=P(rho_c).
    Returns (M_total [g], R_surface [cm]); optionally the (r,P,m) profile.
    M_BH <= 0 delegates to the centre-start solver (exact no-BH limit).
    """
    if M_BH <= 0.0:
        return solve_star(rho_c, eos, h=h, rho_stop=rho_stop,
                          relativistic=relativistic, return_profile=return_profile)

    rhs = tov_rhs if relativistic else newtonian_rhs
    P_stop = eos.P_surface if rho_stop is None else eos.pressure(rho_stop)

    R_BH = schwarzschild_radius(M_BH)
    r = R_BH * (1.0 + eps)                      # start just OUTSIDE the horizon
    P = eos.pressure(rho_c)
    m = M_BH + (4.0 * np.pi / 3.0) * (r**3 - R_BH**3) * rho_c   # BH + thin shell

    # near the horizon dP/dr is very steep; ramp the step up from a small
    # fraction of (r - R_BH) to the requested h.
    h_local = min(h, 0.05 * (r - R_BH)) if h_ramp else h

    rs, Ps, ms = [r], [P], [m]
    for _ in range(max_steps):
        P_new, m_new = rk4_step(r, P, m, h_local, eos, rhs)
        if P_new <= P_stop:
            frac = (P - P_stop) / (P - P_new)
            R = r + frac * h_local
            M = m + frac * (m_new - m)
            if return_profile:
                rs.append(R); Ps.append(P_stop); ms.append(M)
                return M, R, (np.array(rs), np.array(Ps), np.array(ms))
            return M, R
        r += h_local; P, m = P_new, m_new
        rs.append(r); Ps.append(P); ms.append(m)
        if h_ramp and h_local < h:
            h_local = min(h, 1.5 * h_local)
    if return_profile:
        return np.nan, np.nan, (np.array(rs), np.array(Ps), np.array(ms))
    return np.nan, np.nan


# ---------------------------------------------------------------------------
# Route B: consumption-radius construction (recommended for the ejection study).
#          Build the ordinary star once, then "remove the core" out to r_b where
#          the enclosed gravitational mass first reaches M_BH.
# ---------------------------------------------------------------------------
def solve_star_consumed(M_BH, rho_c, eos, h=2.0e3, rho_stop=None,
                        return_profile=False):
    """
    Returns (M_total, R_surface, r_b, M_env) for a star of central density rho_c
    whose inner region (r < r_b) has been swallowed by a BH of mass M_BH.

    M_total = full ADM mass of the original star (unchanged: the BH is built
              from matter that was already there).
    M_env   = gravitational mass still outside the BH = M_total - M_BH.
    r_b     = consumption radius (where enclosed mass == M_BH).
    Returns NaNs if M_BH exceeds the star's total mass.
    """
    M_tot, R_surf, (rs, Ps, ms) = solve_star(
        rho_c, eos, h=h, rho_stop=rho_stop, return_profile=True)

    if not np.isfinite(M_tot) or M_BH >= M_tot:
        if return_profile:
            return np.nan, np.nan, np.nan, np.nan, (rs, Ps, ms)
        return np.nan, np.nan, np.nan, np.nan

    # first radius at which enclosed mass reaches M_BH (linear interp)
    j = np.searchsorted(ms, M_BH)
    if j == 0:
        r_b = rs[0]
    else:
        f = (M_BH - ms[j - 1]) / (ms[j] - ms[j - 1])
        r_b = rs[j - 1] + f * (rs[j] - rs[j - 1])

    M_env = M_tot - M_BH
    if return_profile:
        keep = rs >= r_b
        return M_tot, R_surf, r_b, M_env, (rs[keep], Ps[keep], ms[keep])
    return M_tot, R_surf, r_b, M_env


if __name__ == "__main__":
    # quick self-test against a built-in polytrope (no data files needed)
    from eos import EOS
    eos = EOS.polytrope()
    rho_c = 1.6e15
    print("no-BH :", solve_star(rho_c, eos))
    print("routeA:", solve_star_with_pbh(0.1 * M_SUN, rho_c, eos))
    print("routeB:", solve_star_consumed(0.1 * M_SUN, rho_c, eos))
