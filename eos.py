"""
eos.py  --  Equation of state for the TOV solver (built from scratch).

Modes of construction:
  1. CompOSE data  (from_compose / from_table / from_name)
  2. analytic polytrope  (polytrope)  -- for testing the machinery

Interpolation backend, selectable per EoS via  interp=:
    "numpy"  -> np.interp     (linear, CLAMPS at the table edges)   [default]
    "scipy"  -> interp1d      (linear, EXTRAPOLATES beyond the edges)
Choose "scipy" to match a code that uses scipy.interpolate.interp1d. SciPy is
imported only when that mode is requested, so "numpy" mode needs no SciPy.

Everything is CGS:  rho [g/cm^3] (mass-energy density),  P [dyn/cm^2].
"""

import numpy as np

c   = 2.99792458e10        # speed of light [cm/s]
m_n = 1.674927498e-24      # neutron mass [g]

MeVfm3_to_dyn = 1.602176634e33   # 1 MeV/fm^3  -> dyn/cm^2
fm3_to_cm3    = 1.0e39           # 1 fm^-3     -> cm^-3


class EOS:
    """Equation of state object exposing pressure(rho) and density(P)."""

    def __init__(self):
        self.rho = None        # density grid  [g/cm^3], increasing
        self.P = None          # pressure grid [dyn/cm^2], increasing
        self.mode = None
        self.interp = "numpy"  # "numpy" (clamps) or "scipy" (extrapolates)
        self.nb = None         # baryon number density grid [fm^-3] (for M_baryonic)
        self._P_of_rho = None  # built only in scipy mode
        self._rho_of_P = None

    # ------------------------------------------------------------------
    # MODE 1 -- real data from CompOSE raw files
    # ------------------------------------------------------------------
    @classmethod
    def from_compose(cls, nb_file="eos.nb", thermo_file="eos.thermo",
                     interp="numpy"):
        """
        Build the EOS from CompOSE eos.nb and eos.thermo (1-D tables).
            thermo col 3 = Q1 = p / n_b          ; thermo col 9 = Q7 = e/(n_b m_n)-1
            rho = (Q7 + 1) * n_b * m_n   ;   P = Q1 * n_b   (then to CGS)
        """
        self = cls()
        self.mode = "compose"
        nb_raw = np.loadtxt(nb_file)
        n_points = int(nb_raw[1])
        nb = nb_raw[2:2 + n_points]                  # [fm^-3]
        thermo = np.loadtxt(thermo_file, skiprows=1)
        Q1 = thermo[:, 3]
        Q7 = thermo[:, 9]
        nb_cgs = nb * fm3_to_cm3
        P   = (Q1 * nb) * MeVfm3_to_dyn               # [dyn/cm^2]
        rho = (Q7 + 1.0) * nb_cgs * m_n               # [g/cm^3]
        self._finalize(rho, P, nb=nb, interp=interp)
        return self

    # ------------------------------------------------------------------
    # MODE 1b -- reduced 1-D cold beta-equilibrium table (from build_eos.py)
    # ------------------------------------------------------------------
    @classmethod
    def from_table(cls, path="eos_cold_beta.dat", interp="numpy"):
        """Load a pre-reduced table with columns nb, rho[g/cm^3], P[dyn/cm^2]."""
        self = cls()
        self.mode = "compose"
        d = np.loadtxt(path)
        self._finalize(d[:, 1], d[:, 2], nb=d[:, 0], interp=interp)
        return self

    @classmethod
    def from_name(cls, name, tables_dir="eos_tables", interp="numpy"):
        """Load eos_tables/<name>.dat (the standard prepared table)."""
        import os
        return cls.from_table(os.path.join(tables_dir, f"{name}.dat"),
                              interp=interp)

    # ------------------------------------------------------------------
    # MODE 2 -- analytic polytrope for testing (no data needed)
    # ------------------------------------------------------------------
    @classmethod
    def polytrope(cls, K=1.2e5, Gamma=2.0,
                  rho_min=1.0e4, rho_max=5.0e15, n=4000, interp="numpy"):
        """Toy EOS  P = K * rho^Gamma.  Used only to test the solver."""
        self = cls()
        self.mode = "polytrope"
        self.K = K
        self.Gamma = Gamma
        rho = np.logspace(np.log10(rho_min), np.log10(rho_max), n)
        P = K * rho**Gamma
        self._finalize(rho, P, interp=interp)
        return self

    # ------------------------------------------------------------------
    # shared setup
    # ------------------------------------------------------------------
    def _finalize(self, rho, P, nb=None, interp="numpy"):
        good = (rho > 0.0) & (P > 0.0) & np.isfinite(rho) & np.isfinite(P)
        rho, P = rho[good], P[good]
        if nb is not None:
            nb = np.asarray(nb)[good]
        idx = np.argsort(rho)                         # ascending density
        self.rho = rho[idx]
        self.P = P[idx]
        self.nb = nb[idx] if nb is not None else None
        if not np.all(np.diff(self.P) > 0):
            raise ValueError("Pressure grid is not strictly increasing.")

        if interp not in ("numpy", "scipy"):
            raise ValueError(f"interp must be 'numpy' or 'scipy', got {interp!r}")
        self.interp = interp
        if interp == "scipy":
            from scipy.interpolate import interp1d     # imported only on demand
            self._P_of_rho = interp1d(self.rho, self.P, kind="linear",
                                      fill_value="extrapolate")
            self._rho_of_P = interp1d(self.P, self.rho, kind="linear",
                                      fill_value="extrapolate")

    # ------------------------------------------------------------------
    # lookups
    # ------------------------------------------------------------------
    def pressure(self, rho):
        if self.mode == "polytrope":
            return self.K * rho**self.Gamma
        if self.interp == "scipy":
            return float(self._P_of_rho(rho))
        return np.interp(rho, self.rho, self.P)        # numpy: clamps at edges

    def density(self, P):
        if self.mode == "polytrope":
            Pc = P if P > 0.0 else 0.0
            return max((Pc / self.K)**(1.0 / self.Gamma), 1.0e-12)
        if self.interp == "scipy":
            return float(self._rho_of_P(P))
        return np.interp(P, self.P, self.rho)          # numpy: clamps at edges

    @property
    def P_surface(self):
        """Lowest tabulated pressure -- a natural 'surface' threshold."""
        return self.P[0]

    def rest_mass_density(self, P):
        """
        Baryon rest-mass density [g/cm^3] at pressure P:  m_n * n_b.
        This is what the baryonic-mass integral uses (NOT the mass-energy
        density rho, which also carries internal energy). Needs the nb column.
        """
        if self.nb is None:
            raise ValueError("This EOS has no baryon-density (nb) column.")
        nb = np.interp(P, self.P, self.nb)        # fm^-3
        return m_n * nb * fm3_to_cm3              # g/cm^3

    def summary(self):
        print(f"EOS mode : {self.mode}  (interp={self.interp})")
        print(f"points   : {len(self.rho)}")
        print(f"rho range: {self.rho.min():.3e} .. {self.rho.max():.3e} g/cm^3")
        print(f"P range  : {self.P.min():.3e} .. {self.P.max():.3e} dyn/cm^2")
