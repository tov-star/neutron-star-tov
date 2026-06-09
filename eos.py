"""
eos.py  --  Equation of state for the TOV solver (built from scratch).

Two modes:

  1. CompOSE mode: read the APR tables `eos.nb` and `eos.thermo`
     downloaded from https://compose.obspm.fr/eos/297 and build
     P(rho) and rho(P) by interpolation.

  2. Polytrope mode: an analytic toy EOS  P = K * rho^Gamma  used to
     test the solver machinery when no data file is available.

Everything is in CGS:
    rho : g/cm^3   (mass-energy density)
    P   : dyn/cm^2

We deliberately use only numpy (np.interp) for the table lookup instead
of scipy, so the interpolation step is fully transparent / "from scratch".
"""

import numpy as np

# ----------------------------------------------------------------------
# Physical constants (CGS) -- matched to the teammate code for fair
# comparison.
# ----------------------------------------------------------------------
c   = 2.99792458e10        # speed of light [cm/s]
m_n = 1.674927498e-24      # neutron mass [g]

# Unit conversions out of CompOSE's natural units
MeVfm3_to_dyn = 1.602176634e33   # 1 MeV/fm^3  -> dyn/cm^2
fm3_to_cm3    = 1.0e39           # 1 fm^-3     -> cm^-3


class EOS:
    """Equation of state object exposing pressure(rho) and density(P)."""

    def __init__(self):
        self.rho = None        # density grid  [g/cm^3], increasing
        self.P = None          # pressure grid [dyn/cm^2], increasing
        self.mode = None

    # ------------------------------------------------------------------
    # MODE 1 -- real APR data from CompOSE
    # ------------------------------------------------------------------
    @classmethod
    def from_compose(cls, nb_file="eos.nb", thermo_file="eos.thermo"):
        """
        Build the EOS from CompOSE `eos.nb` and `eos.thermo`.

        Column convention (CompOSE manual p.37):
            thermo col index 3  = Q1 = p / n_b           [MeV]   (per baryon)
            thermo col index 9  = Q7 = e/(n_b m_n c^2)-1 [dimensionless]

        Mass-energy density (what gravitates in GR):
            rho = (Q7 + 1) * n_b * m_n
        Pressure:
            P   = Q1 * n_b      (then convert MeV/fm^3 -> dyn/cm^2)
        """
        self = cls()
        self.mode = "compose"

        # --- eos.nb: line 0 is a flag, line 1 is the number of points,
        #     then the baryon-number-density grid in fm^-3 ---
        nb_raw = np.loadtxt(nb_file)
        n_points = int(nb_raw[1])
        nb = nb_raw[2:2 + n_points]                  # [fm^-3]

        # --- eos.thermo: skip the first header line ---
        thermo = np.loadtxt(thermo_file, skiprows=1)
        Q1 = thermo[:, 3]                            # p / n_b   [MeV]
        Q7 = thermo[:, 9]                            # scaled internal energy

        nb_cgs = nb * fm3_to_cm3                      # [cm^-3]

        P   = (Q1 * nb) * MeVfm3_to_dyn               # [dyn/cm^2]
        rho = (Q7 + 1.0) * nb_cgs * m_n               # [g/cm^3]

        self._finalize(rho, P)
        return self

    # ------------------------------------------------------------------
    # MODE 1b -- reduced 1-D cold beta-equilibrium table (from build_eos.py)
    # ------------------------------------------------------------------
    @classmethod
    def from_table(cls, path="eos_cold_beta.dat"):
        """Load a pre-reduced table with columns nb, rho[g/cm^3], P[dyn/cm^2]."""
        self = cls()
        self.mode = "compose"
        d = np.loadtxt(path)
        self._finalize(d[:, 1], d[:, 2])
        return self

    # ------------------------------------------------------------------
    # MODE 2 -- analytic polytrope for testing (no data needed)
    # ------------------------------------------------------------------
    @classmethod
    def polytrope(cls, K=1.2e5, Gamma=2.0,
                  rho_min=1.0e4, rho_max=5.0e15, n=4000):
        """
        Toy EOS  P = K * rho^Gamma.  Used only to test the solver.
        Defaults give a neutron-star-scale object (M ~ 1-2 Msun, R ~ 10-15 km).
        """
        self = cls()
        self.mode = "polytrope"
        self.K = K
        self.Gamma = Gamma
        rho = np.logspace(np.log10(rho_min), np.log10(rho_max), n)
        P = K * rho**Gamma
        self._finalize(rho, P)
        return self

    # ------------------------------------------------------------------
    # shared setup
    # ------------------------------------------------------------------
    def _finalize(self, rho, P):
        mask = (rho > 0.0) & (P > 0.0) & np.isfinite(rho) & np.isfinite(P)
        rho, P = rho[mask], P[mask]
        idx = np.argsort(rho)                         # ascending density
        self.rho = rho[idx]
        self.P = P[idx]
        # both grids must be monotonic for np.interp to be valid
        if not np.all(np.diff(self.P) > 0):
            raise ValueError("Pressure grid is not strictly increasing.")

    # ------------------------------------------------------------------
    # lookups  (np.interp clamps to the table ends -- no wild extrapolation)
    # ------------------------------------------------------------------
    def pressure(self, rho):
        if self.mode == "polytrope":
            return self.K * rho**self.Gamma
        return np.interp(rho, self.rho, self.P)

    def density(self, P):
        if self.mode == "polytrope":
            # clamp to >=0 (RK4 intermediate stages can probe P<0 near the
            # surface) and floor the density so the TOV terms never divide
            # by zero.
            Pc = P if P > 0.0 else 0.0
            return max((Pc / self.K)**(1.0 / self.Gamma), 1.0e-12)
        return np.interp(P, self.P, self.rho)

    @property
    def P_surface(self):
        """Lowest tabulated pressure -- a natural 'surface' threshold."""
        return self.P[0]

    def summary(self):
        print(f"EOS mode : {self.mode}")
        print(f"points   : {len(self.rho)}")
        print(f"rho range: {self.rho.min():.3e} .. {self.rho.max():.3e} g/cm^3")
        print(f"P range  : {self.P.min():.3e} .. {self.P.max():.3e} dyn/cm^2")
