"""
accretion_survey.py  --  Part 2: the accretion relation + spin-up / ejection.

Survey the (M_BH, rho_c) plane. For each pair we build the consumed-core star
(pbh_solver.solve_star_consumed: a star of central density rho_c whose inner
region out to r_b has been swallowed by a BH of mass M_BH) and then evaluate the
rotational mass-shedding (Kepler) criterion using an initial spin period P0 taken
from the millisecond-pulsar population.

WHAT EACH AXIS / CURVE MEANS
  - color  = M_total(M_BH, rho_c), the star's gravitational (ADM) mass.
  - constant-M contours = accretion tracks. In the consumed-core model the total
    mass is set by rho_c alone (swapping the core for a BH of equal enclosed mass
    leaves the exterior metric unchanged), so a track is a horizontal cut: pick an
    observed NS mass, then march M_BH to the right and watch the geometry evolve.
  - ejection boundary = the locus where the spun-up equatorial velocity first
    reaches the Kepler velocity (v_eq = v_K). To its right the star sheds matter.

ROTATION MODEL  (Fuller, Kusenko & Takhistov 2017, Eqs. 13-15 + main text)
  R_S  = 2 G M_BH / c^2                         BH Schwarzschild radius
  R_p  = R_NS - r_b + R_S                        new polar radius (core removed)
  R_eq = (3/2) R_p                               Roche mass-shedding equatorial radius
  Omega0 = 2 pi / P0 ;  R_eq0 = (3/2) R_NS ;  v_eq0 = Omega0 R_eq0
  v_eq(r_b) = v_eq0 * R_eq0 / R_eq               specific ang. mom. of equatorial
                                                 element conserved (v_eq * R_eq = const)
  v_K(R_eq) = sqrt(G M_total / R_eq)             Kepler / mass-shedding velocity
  EJECTION  when  v_eq >= v_K.

These are deliberately simple, transparent prescriptions; they reproduce the
order-of-magnitude result that ~0.3-0.4 Msun of consumed core spins a millisecond
pulsar up to mass shedding.
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
from pbh_solver import solve_star_consumed, schwarzschild_radius

G = 6.67430e-8
c = 2.99792458e10
DEFAULT = "APR"

# representative BH masses from the project sheet (solar masses)
MBH_GRID = np.array([0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40])
# initial spin period of the millisecond-pulsar population (s)
P0_MSP = 1.5e-3


def ejection_state(M_tot, R_NS, r_b, M_BH, P0):
    """Return (R_eq [cm], v_eq, v_K, ratio=v_eq/v_K). NaN if BH ate the star."""
    R_S = schwarzschild_radius(M_BH)
    R_p = R_NS - r_b + R_S
    if not np.isfinite(R_p) or R_p <= 0:
        return np.nan, np.nan, np.nan, np.nan
    R_eq = 1.5 * R_p
    Omega0 = 2.0 * np.pi / P0
    R_eq0 = 1.5 * R_NS
    v_eq = (Omega0 * R_eq0) * R_eq0 / R_eq          # = v_eq0 * R_eq0 / R_eq
    v_K = np.sqrt(G * M_tot / R_eq)
    return R_eq, v_eq, v_K, v_eq / v_K


def run_survey(eos, mbh_grid=MBH_GRID, rho_c_grid=None, P0=P0_MSP,
               h=2.0e3, rho_stop=None):
    if rho_c_grid is None:
        rho_c_grid = np.logspace(np.log10(4.0e14),
                                 np.log10(eos.rho[-1] * 0.98), 40)
    nB, nR = len(mbh_grid), len(rho_c_grid)
    M = np.full((nR, nB), np.nan)   # total mass [Msun]
    Rb = np.full((nR, nB), np.nan)  # consumption radius [km]
    Req = np.full((nR, nB), np.nan)
    ratio = np.full((nR, nB), np.nan)
    for j, rc in enumerate(rho_c_grid):
        for i, mbh in enumerate(mbh_grid):
            Mt, Rs, r_b, Menv = solve_star_consumed(mbh * M_SUN, rc, eos,
                                                    h=h, rho_stop=rho_stop)
            if not np.isfinite(Mt):
                continue
            M[j, i] = Mt / M_SUN
            Rb[j, i] = r_b / 1e5
            R_eq, v_eq, v_K, rr = ejection_state(Mt, Rs, r_b, mbh * M_SUN, P0)
            Req[j, i] = R_eq / 1e5 if np.isfinite(R_eq) else np.nan
            ratio[j, i] = rr
    return dict(mbh=mbh_grid, rho_c=rho_c_grid, M=M, Rb=Rb, Req=Req, ratio=ratio)


def plot_survey(S, name, P0=P0_MSP, outdir=None):
    outdir = outdir or os.path.join("results", name)
    os.makedirs(outdir, exist_ok=True)
    mbh, rho_c = S["mbh"], S["rho_c"]
    X, Y = np.meshgrid(mbh, rho_c)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.4))

    # Panel A: color = M_total, constant-M contours, ejection boundary
    pcA = axA.pcolormesh(X, Y, S["M"], shading="auto", cmap="viridis")
    fig.colorbar(pcA, ax=axA, label=r"$M_{\rm total}$  [$M_\odot$]")
    levels = np.round(np.nanpercentile(S["M"], [10, 30, 50, 70, 90]), 2)
    cs = axA.contour(X, Y, S["M"], levels=np.unique(levels),
                     colors="white", linewidths=1.0)
    axA.clabel(cs, fmt="%.2f", fontsize=8)
    # ejection boundary: ratio = 1
    if np.isfinite(S["ratio"]).any():
        axA.contour(X, Y, S["ratio"], levels=[1.0],
                    colors="crimson", linewidths=2.5)
        axA.plot([], [], color="crimson", lw=2.5,
                 label=f"ejection ($v_{{eq}}=v_K$), $P_0$={P0*1e3:.1f} ms")
        axA.legend(loc="upper right", fontsize=8)
    axA.set_yscale("log")
    axA.set_xlabel(r"$M_{\rm BH}$  [$M_\odot$]")
    axA.set_ylabel(r"$\rho_c$  [g/cm$^3$]")
    axA.set_title("Accretion relation: total mass + ejection threshold")

    # Panel B: spin-up ratio v_eq/v_K (this is where the 2-D structure lives)
    pcB = axB.pcolormesh(X, Y, np.clip(S["ratio"], 0, 2.0),
                         shading="auto", cmap="RdBu_r", vmin=0, vmax=2)
    fig.colorbar(pcB, ax=axB, label=r"$v_{eq}/v_K$  (>1 ejects)")
    axB.contour(X, Y, S["ratio"], levels=[1.0], colors="black", linewidths=2.0)
    axB.set_yscale("log")
    axB.set_xlabel(r"$M_{\rm BH}$  [$M_\odot$]")
    axB.set_ylabel(r"$\rho_c$  [g/cm$^3$]")
    axB.set_title(f"Spin-up ratio (MSP, $P_0$={P0*1e3:.1f} ms)")

    fig.suptitle(f"{name}: PBH accretion survey")
    fig.tight_layout()
    f = os.path.join(outdir, "accretion_Mmap.png")
    fig.savefig(f, dpi=130); plt.close(fig)
    return f


def trace_fixed_mass(S, eos, name, target_M=1.4, P0=P0_MSP,
                     h=2.0e3, rho_stop=None, outdir=None):
    """Pick rho_c giving total mass ~ target_M; trace R_eq, spin period, ratio
    vs a fine M_BH grid; mark the ejection onset."""
    outdir = outdir or os.path.join("results", name)
    os.makedirs(outdir, exist_ok=True)
    # find rho_c for target mass from the no-BH M(rho_c)
    rc_scan = np.logspace(np.log10(4e14), np.log10(eos.rho[-1] * 0.98), 80)
    Ms = np.array([solve_star(rc, eos, h=h, rho_stop=rho_stop)[0] / M_SUN
                   for rc in rc_scan])
    rc = rc_scan[np.nanargmin(np.abs(Ms - target_M))]
    M_here = solve_star(rc, eos, h=h, rho_stop=rho_stop)[0] / M_SUN

    mbh_fine = np.linspace(0.0, 0.45, 46)
    Req, P_spin, rr, rbs = [], [], [], []
    for mbh in mbh_fine:
        Mt, Rs, r_b, Menv = solve_star_consumed(mbh * M_SUN, rc, eos,
                                                h=h, rho_stop=rho_stop)
        if not np.isfinite(Mt):
            Req.append(np.nan); P_spin.append(np.nan); rr.append(np.nan); rbs.append(np.nan); continue
        R_eq, v_eq, v_K, ratio = ejection_state(Mt, Rs, r_b, mbh * M_SUN, P0)
        Req.append(R_eq / 1e5 if np.isfinite(R_eq) else np.nan)
        Omega = v_eq / R_eq if np.isfinite(R_eq) else np.nan
        P_spin.append(2 * np.pi / Omega * 1e3 if np.isfinite(Omega) else np.nan)  # ms
        rr.append(ratio); rbs.append(r_b / 1e5)
    Req = np.array(Req); rr = np.array(rr); P_spin = np.array(P_spin)

    onset = None
    idx = np.where(rr >= 1.0)[0]
    if len(idx):
        onset = mbh_fine[idx[0]]

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    ax.plot(mbh_fine, Req, "-o", ms=3, color="navy", label=r"$R_{eq}$ [km]")
    ax.set_xlabel(r"$M_{\rm BH}$ [$M_\odot$]")
    ax.set_ylabel(r"$R_{eq}$ [km]", color="navy")
    ax2 = ax.twinx()
    ax2.plot(mbh_fine, rr, "-s", ms=3, color="crimson", label=r"$v_{eq}/v_K$")
    ax2.axhline(1.0, color="crimson", ls="--", lw=1)
    ax2.set_ylabel(r"$v_{eq}/v_K$", color="crimson")
    if onset is not None:
        ax.axvline(onset, color="k", ls=":", lw=1.5)
        ax.text(onset, ax.get_ylim()[1] * 0.9,
                f" ejects at\n $M_{{BH}}\\approx{onset:.2f}$", fontsize=9)
    ax.set_title(f"{name}: M={M_here:.2f} Msun track, P0={P0*1e3:.1f} ms")
    fig.tight_layout()
    f = os.path.join(outdir, "accretion_track.png")
    fig.savefig(f, dpi=130); plt.close(fig)
    return f, rc, M_here, onset, mbh_fine, Req, rr


def main(name=DEFAULT, P0=P0_MSP):
    eos = EOS.from_name(name)
    outdir = os.path.join("results", name)
    os.makedirs(outdir, exist_ok=True)
    S = run_survey(eos, P0=P0)
    np.savez(os.path.join(outdir, "accretion_grid.npz"), **S)
    f1 = plot_survey(S, name, P0=P0, outdir=outdir)
    f2, rc, M_here, onset, *_ = trace_fixed_mass(S, eos, name, 1.4, P0, outdir=outdir)
    print(f"saved {f1}")
    print(f"saved {f2}")
    if onset is not None:
        print(f"M={M_here:.2f} Msun, P0={P0*1e3:.1f} ms -> ejection at M_BH~{onset:.2f} Msun")
    else:
        print(f"M={M_here:.2f} Msun, P0={P0*1e3:.1f} ms -> no ejection in range")


if __name__ == "__main__":
    nm = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    main(nm)
