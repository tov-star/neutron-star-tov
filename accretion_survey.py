"""
accretion_survey.py  --  Part 2: accretion relation + spin-up / ejection.
PRODUCTION-READY: parallel, checkpoint/resume, configurable grid.

Survey the (M_BH, rho_c) plane on the APR equation of state. For each rho_c the
star is solved ONCE (full profile); every M_BH then reuses that profile to find
the consumption radius r_b (enclosed mass == M_BH) and evaluate the rotational
mass-shedding (Kepler) criterion for an initial spin period P0.

SPIN-UP MODELS (both are computed and stored)
  specific_j  conservative: equatorial element conserves v_eq*R_eq (no transfer).
  rigid       efficient transfer (Kusenko): rigid body, J=I*Omega conserved, with
              I from the actual profile, I_env scaled by (R_p/R_NS)^2 (homologous
              envelope contraction). Differs from specific_j only by I0/I_env.

GEOMETRY / KEPLER (Fuller, Kusenko & Takhistov 2017)
  R_S=2GM_BH/c^2 ; R_p=R_NS-r_b+R_S ; R_eq=b*R_p (b=3/2 Roche) ;
  v_K=sqrt(G M_tot/R_eq) ; EJECTION when v_eq>=v_K.

RUN
  python accretion_survey.py                       # APR, default grid, all cores
  python accretion_survey.py APR --n-rho 200 --n-mbh 200 --nproc 16
  python accretion_survey.py APR --resume          # continue an interrupted run
  python accretion_survey.py APR --fresh           # ignore old checkpoints
Each completed rho_c row is checkpointed to <outdir>/checkpoints/row_XXXX.npz, so
the job can be killed and restarted without losing finished work. The per-cell
spin model is cheap now; the parallel+checkpoint scaffolding is what you keep when
the per-model physics (rotating equilibrium / dynamical ejection) gets expensive.
"""

import os, sys, glob, re, time, argparse, traceback
import multiprocessing as mp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eos import EOS
from solver import solve_star, M_SUN
from pbh_solver import schwarzschild_radius

G = 6.67430e-8
c = 2.99792458e10
BASE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = "APR"

MBH_REPR = np.array([0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40])
P0_MSP = 6.4e-3
OBLATENESS = 1.5
ROWKEYS = ("M", "Rb", "Req", "ratio_sj", "ratio_rg")


# ---------------------------------------------------------------------------
# structure helpers  (version-safe trapezoid; np.trapz removed in NumPy 2.0)
# ---------------------------------------------------------------------------
def _trapz(y, x):
    y = np.asarray(y, float); x = np.asarray(x, float)
    if y.size < 2:
        return 0.0
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def moment_of_inertia(rs, rhos, r_lo, r_hi):
    """(8 pi / 3) * integral_{r_lo}^{r_hi} rho r^4 dr  (Newtonian)."""
    m = (rs >= r_lo) & (rs <= r_hi)
    return (8.0 * np.pi / 3.0) * _trapz(rhos[m] * rs[m]**4, rs[m])


def consumption_radius(rs, ms, M_BH):
    j = int(np.searchsorted(ms, M_BH))
    if j <= 0:
        return rs[0]
    if j >= ms.size:
        return rs[-1]
    f = (M_BH - ms[j - 1]) / (ms[j] - ms[j - 1])
    return rs[j - 1] + f * (rs[j] - rs[j - 1])


def spin_state(M_tot, R_NS, r_b, M_BH, rs, rhos, I0, P0, mode, b):
    R_S = schwarzschild_radius(M_BH)
    R_p = R_NS - r_b + R_S
    if not np.isfinite(R_p) or R_p <= 0:
        return dict(R_eq=np.nan, ratio=np.nan)
    R_eq = b * R_p
    Omega0 = 2.0 * np.pi / P0
    if mode == "specific_j":
        R_eq0 = b * R_NS
        v_eq = Omega0 * R_eq0 * R_eq0 / R_eq
    elif mode == "rigid":
        I_env = moment_of_inertia(rs, rhos, r_b, R_NS) * (R_p / R_NS)**2
        if I_env <= 0:
            return dict(R_eq=R_eq, ratio=np.nan)
        v_eq = (I0 * Omega0 / I_env) * R_eq
    else:
        raise ValueError(mode)
    v_K = np.sqrt(G * M_tot / R_eq)
    return dict(R_eq=R_eq, ratio=v_eq / v_K)


# ---------------------------------------------------------------------------
# one row of the survey (= one rho_c, all M_BH). Top-level + worker globals so
# it is picklable for multiprocessing.
# ---------------------------------------------------------------------------
_W = {}


def _init_worker(eos_name, mbh_grid, P0, b, h, rho_stop):
    _W["eos"] = EOS.from_name(eos_name)
    _W["mbh"] = mbh_grid; _W["P0"] = P0; _W["b"] = b
    _W["h"] = h; _W["rho_stop"] = rho_stop


def _row_for(rc, eos, mbh, P0, b, h, rho_stop):
    row = {k: np.full(mbh.size, np.nan) for k in ROWKEYS}
    M_tot, R_NS, (rs, Ps, ms) = solve_star(
        rc, eos, h=h, rho_stop=rho_stop, return_profile=True)
    if not np.isfinite(M_tot):
        return row
    rhos = eos.density(Ps)
    I0 = moment_of_inertia(rs, rhos, 0.0, R_NS)
    for i, mb in enumerate(mbh):
        M_BH = mb * M_SUN
        if M_BH >= M_tot:
            continue
        r_b = consumption_radius(rs, ms, M_BH)
        sj = spin_state(M_tot, R_NS, r_b, M_BH, rs, rhos, I0, P0, "specific_j", b)
        rg = spin_state(M_tot, R_NS, r_b, M_BH, rs, rhos, I0, P0, "rigid", b)
        row["M"][i] = M_tot / M_SUN
        row["Rb"][i] = r_b / 1e5
        row["Req"][i] = sj["R_eq"] / 1e5 if np.isfinite(sj["R_eq"]) else np.nan
        row["ratio_sj"][i] = sj["ratio"]
        row["ratio_rg"][i] = rg["ratio"]
    return row


def _compute_row(task):
    """task=(j, rho_c) -> (j, row_dict). Errors are isolated per row."""
    j, rc = task
    try:
        row = _row_for(rc, _W["eos"], _W["mbh"], _W["P0"], _W["b"],
                       _W["h"], _W["rho_stop"])
    except Exception:
        sys.stderr.write(f"[row {j}] failed:\n{traceback.format_exc()}\n")
        row = {k: np.full(_W["mbh"].size, np.nan) for k in ROWKEYS}
    return j, row


# ---------------------------------------------------------------------------
# parallel driver with checkpoint / resume
# ---------------------------------------------------------------------------
def _grids_match(meta, mbh_grid, rho_c_grid):
    return (meta["mbh"].shape == mbh_grid.shape
            and meta["rho_c"].shape == rho_c_grid.shape
            and np.allclose(meta["mbh"], mbh_grid)
            and np.allclose(meta["rho_c"], rho_c_grid))


def run_survey(eos_name, mbh_grid, rho_c_grid, P0, b, h, rho_stop,
               nproc, outdir, resume=True):
    ckpt = os.path.join(outdir, "checkpoints")
    os.makedirs(ckpt, exist_ok=True)
    meta_path = os.path.join(outdir, "meta.npz")

    if resume and os.path.exists(meta_path):
        meta = np.load(meta_path)
        if not _grids_match(meta, mbh_grid, rho_c_grid):
            sys.exit("Grid in existing checkpoints differs from requested grid. "
                     "Use --fresh to discard, or match the previous resolution.")
    np.savez(meta_path, mbh=mbh_grid, rho_c=rho_c_grid,
             P0=P0, b=b, h=h)

    nR = rho_c_grid.size
    done = set()
    if resume:
        for fn in glob.glob(os.path.join(ckpt, "row_*.npz")):
            done.add(int(re.search(r"row_(\d+)", fn).group(1)))
    tasks = [(j, rho_c_grid[j]) for j in range(nR) if j not in done]
    print(f"grid {nR} x {mbh_grid.size}  |  {len(done)} rows cached, "
          f"{len(tasks)} to compute  |  nproc={nproc}")

    t0 = time.time()
    if tasks:
        with mp.Pool(nproc, initializer=_init_worker,
                     initargs=(eos_name, mbh_grid, P0, b, h, rho_stop)) as pool:
            for k, (j, row) in enumerate(pool.imap_unordered(_compute_row, tasks), 1):
                np.savez(os.path.join(ckpt, f"row_{j:04d}.npz"), **row)
                el = time.time() - t0
                eta = el / k * (len(tasks) - k)
                print(f"  row {j:4d}  [{k}/{len(tasks)}]  "
                      f"elapsed {el:6.1f}s  eta {eta:6.1f}s", flush=True)

    # assemble full grid from checkpoints
    out = {key: np.full((nR, mbh_grid.size), np.nan) for key in ROWKEYS}
    for fn in glob.glob(os.path.join(ckpt, "row_*.npz")):
        j = int(re.search(r"row_(\d+)", fn).group(1))
        d = np.load(fn)
        for key in ROWKEYS:
            out[key][j] = d[key]
    out["mbh"], out["rho_c"] = mbh_grid, rho_c_grid
    np.savez(os.path.join(outdir, "accretion_grid.npz"), **out)
    return out


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------
def plot_survey(S, name, P0, outdir):
    X, Y = np.meshgrid(S["mbh"], S["rho_c"])
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.2))
    pc = axs[0].pcolormesh(X, Y, S["M"], shading="auto", cmap="viridis")
    fig.colorbar(pc, ax=axs[0], label=r"$M_{\rm total}$ [$M_\odot$]")
    lv = np.unique(np.round(np.nanpercentile(S["M"], [10, 30, 50, 70, 90]), 2))
    if lv.size:
        cs = axs[0].contour(X, Y, S["M"], levels=lv, colors="white", linewidths=1.0)
        axs[0].clabel(cs, fmt="%.2f", fontsize=8)
    axs[0].set_title("Total mass + accretion tracks")
    for ax, key, ttl in ((axs[1], "ratio_sj", "specific-j (conservative)"),
                         (axs[2], "ratio_rg", "rigid (efficient transfer)")):
        pc = ax.pcolormesh(X, Y, np.clip(S[key], 0, 2), shading="auto",
                           cmap="RdBu_r", vmin=0, vmax=2)
        fig.colorbar(pc, ax=ax, label=r"$v_{eq}/v_K$")
        if np.nanmax(S[key]) >= 1.0:
            ax.contour(X, Y, S[key], levels=[1.0], colors="k", linewidths=2)
        ax.set_title(f"Spin-up: {ttl}")
    for ax in axs:
        ax.set_yscale("log")
        ax.set_xlabel(r"$M_{\rm BH}$ [$M_\odot$]")
        ax.set_ylabel(r"$\rho_c$ [g/cm$^3$]")
    fig.suptitle(f"{name}: PBH accretion survey  (P0 = {P0*1e3:.1f} ms)")
    fig.tight_layout()
    f = os.path.join(outdir, "accretion_Mmap.png")
    fig.savefig(f, dpi=130); plt.close(fig)
    return f


def plot_threshold_vs_period(S, name, P0_ref, target_M, outdir):
    j = int(np.nanargmin(np.abs(S["M"][:, 0] - target_M)))
    periods = np.linspace(0.8e-3, 6.5e-3, 60)
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    for key, lab, col in (("ratio_sj", "specific-j", "navy"),
                          ("ratio_rg", "rigid (eff. transfer)", "crimson")):
        thr = []
        for P0 in periods:
            r = S[key][j, :] * (P0_ref / P0)
            hit = np.where(r >= 1.0)[0]
            if hit.size == 0:
                thr.append(np.nan)
            elif hit[0] == 0:
                thr.append(S["mbh"][0])
            else:
                i = hit[0]
                f = (1.0 - r[i - 1]) / (r[i] - r[i - 1])
                thr.append(S["mbh"][i - 1] + f * (S["mbh"][i] - S["mbh"][i - 1]))
        ax.plot(periods * 1e3, thr, "-", color=col, lw=2, label=lab)
    ax.axhspan(0.3, 0.4, color="gray", alpha=0.15, label="sheet estimate 0.3-0.4")
    ax.set_xlabel(r"initial spin period $P_0$ [ms]")
    ax.set_ylabel(r"ejection threshold $M_{\rm BH}$ [$M_\odot$]")
    ax.set_title(f"{name}: M={S['M'][j,0]:.2f} Msun ejection threshold")
    ax.set_ylim(0, 0.45); ax.legend()
    fig.tight_layout()
    f = os.path.join(outdir, "ejection_threshold.png")
    fig.savefig(f, dpi=130); plt.close(fig)
    return f


# ---------------------------------------------------------------------------
def build_grids(eos, args):
    rho_max = args.rho_max if args.rho_max else eos.rho[-1] * 0.98
    rho_c_grid = np.logspace(np.log10(args.rho_min), np.log10(rho_max), args.n_rho)
    if args.n_mbh:
        mbh_grid = np.linspace(args.mbh_min, args.mbh_max, args.n_mbh)
    else:
        mbh_grid = MBH_REPR
    return mbh_grid, rho_c_grid


def main(argv=None):
    p = argparse.ArgumentParser(description="PBH accretion parameter survey")
    p.add_argument("eos", nargs="?", default=DEFAULT)
    p.add_argument("--n-rho", type=int, default=40)
    p.add_argument("--rho-min", type=float, default=4.0e14)
    p.add_argument("--rho-max", type=float, default=None)
    p.add_argument("--n-mbh", type=int, default=0,
                   help="0 -> use representative list; >0 -> linspace grid")
    p.add_argument("--mbh-min", type=float, default=0.005)
    p.add_argument("--mbh-max", type=float, default=0.40)
    p.add_argument("--P0", type=float, default=P0_MSP * 1e3, help="ms")
    p.add_argument("--oblateness", type=float, default=OBLATENESS)
    p.add_argument("--h", type=float, default=2.0e3)
    p.add_argument("--rho-stop", type=float, default=None)
    p.add_argument("--nproc", type=int, default=max(1, mp.cpu_count()))
    p.add_argument("--target-M", type=float, default=1.4)
    p.add_argument("--outdir", default=None)
    p.add_argument("--fresh", action="store_true", help="ignore old checkpoints")
    args = p.parse_args(argv)

    eos = EOS.from_name(args.eos)
    P0 = args.P0 * 1e-3
    outdir = args.outdir or os.path.join(BASE, "results", args.eos)
    os.makedirs(outdir, exist_ok=True)
    if args.fresh:
        for fn in glob.glob(os.path.join(outdir, "checkpoints", "row_*.npz")):
            os.remove(fn)

    mbh_grid, rho_c_grid = build_grids(eos, args)
    S = run_survey(args.eos, mbh_grid, rho_c_grid, P0, args.oblateness,
                   args.h, args.rho_stop, args.nproc, outdir,
                   resume=not args.fresh)
    f1 = plot_survey(S, args.eos, P0, outdir)
    f2 = plot_threshold_vs_period(S, args.eos, P0, args.target_M, outdir)
    print(f"saved {f1}\nsaved {f2}")
    print(f"max v_eq/v_K : specific_j={np.nanmax(S['ratio_sj']):.3f}  "
          f"rigid={np.nanmax(S['ratio_rg']):.3f}")


if __name__ == "__main__":
    main()
