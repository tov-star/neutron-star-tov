"""
tov.py  --  the structure equations (right-hand sides).

We provide BOTH the Newtonian and the relativistic (TOV) right-hand
sides so you can compare them directly (project guiding question #3).
Each returns (dP/dr, dm/dr) given the current (r, P, m) and the EOS.

State vector:
    P : pressure        [dyn/cm^2]
    m : enclosed mass    [g]
    r : radius           [cm]

Mass continuity (same in both):
    dm/dr = 4*pi*r^2 * rho
(using the mass-energy density rho, since in GR energy gravitates).
"""

import numpy as np

G = 6.67430e-8             # cm^3 g^-1 s^-2
c = 2.99792458e10          # cm/s


def newtonian_rhs(r, P, m, eos):
    """Newtonian hydrostatic equilibrium:  dP/dr = -G m rho / r^2."""
    rho = eos.density(P)
    dmdr = 4.0 * np.pi * r**2 * rho
    dPdr = -G * m * rho / r**2
    return dPdr, dmdr


def tov_rhs(r, P, m, eos):
    """
    Tolman-Oppenheimer-Volkoff equation. The three bracketed factors are
    the GR corrections; each one makes gravity effectively *stronger*
    than Newton, which is why neutron stars have a maximum mass.

        dP/dr = -G m rho / r^2
                * (1 + P/(rho c^2))               # pressure adds to inertia
                * (1 + 4 pi r^3 P /(m c^2))        # pressure sources gravity
                / (1 - 2 G m /(r c^2))             # spacetime curvature
    """
    rho = eos.density(P)
    dmdr = 4.0 * np.pi * r**2 * rho
    dPdr = (-G * m * rho / r**2
            * (1.0 + P / (rho * c**2))
            * (1.0 + 4.0 * np.pi * r**3 * P / (m * c**2))
            / (1.0 - 2.0 * G * m / (r * c**2)))
    return dPdr, dmdr
