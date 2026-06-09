# Neutron Star TOV Solver (SURP 2026)

A from-scratch Tolman-Oppenheimer-Volkoff solver for neutron-star structure,
built to survey many equations of state (EoS). The design principle: every EoS
is reduced once to a standard 1-D table, and all the physics code reads only
that table, so adding a new EoS costs a download and one catalog row.

## Layout

```
neutron-star-tov/
  eos.py  tov.py  rk4.py  solver.py   the engine (EoS-agnostic; never changes per EoS)
  build_eos.py                        reduce a raw CompOSE EoS -> standard 1-D table
  catalog.py  catalog.csv             the registry of EoS + metadata
  single_star.py mr_curve.py          drivers: take an EoS NAME, read its table
  compare_to_compose.py convergence.py
  run_all.py                          batch: prepare + run every EoS in the catalog
  eos_raw/<NAME>/                     raw CompOSE downloads (git-ignored, large)
  eos_tables/<NAME>.dat               reduced 1-D tables (small, COMMIT these)
  results/<NAME>/                     figures + data (git-ignored, regenerable)
```

The standard table (`eos_tables/<NAME>.dat`) has three columns:
`nb[fm^-3]  rho[g/cm^3]  P[dyn/cm^2]`. It is the universal interface between the
EoS-specific parsing (in `build_eos.py`) and the EoS-agnostic solver.

## Adding a new EoS

1. Download it from CompOSE, unzip into `eos_raw/<NAME>/` (needs at least
   `eos.nb` and `eos.thermo`; `eos.mr` enables the reference comparison).
2. Add a row to `catalog.csv`: name, CompOSE id, model, dim, ref_Mmax, ref_R14
   (the reference values come straight off the EoS's CompOSE page).
3. `python build_eos.py <NAME>`  -> writes `eos_tables/<NAME>.dat`.
   It auto-detects 1-D vs 3-D tables (3-D ones are reduced to cold,
   beta-equilibrium matter).
4. `python mr_curve.py <NAME>`   -> the mass-radius curve + validation.

## Running

```bash
python single_star.py <NAME> [rho_c]   # one model + profiles
python mr_curve.py <NAME>               # M-R curve, max mass, validity check
python compare_to_compose.py <NAME>     # overlay the CompOSE eos.mr reference
python convergence.py                   # integrator order check (uses a polytrope)
python run_all.py                       # prepare + run every EoS in the catalog
```
With no NAME the drivers default to ABHT_QMCRMF1.

## EoS in the catalog

- APR (CompOSE eos/68): the real APR EoS (variational, A18+dv+UIX*), 1-D cold
  beta-eq, Mmax 2.19 Msun. Download eos/68 into eos_raw/APR/ to use it.
- ABHT_QMCRMF1 (CompOSE eos/297): a relativistic mean-field EoS, 3-D
  general-purpose, Mmax 1.95 Msun. (NOTE: despite the project sheet, eos/297 is
  NOT APR -- it is ABHT QMC-RMF1.)

## What to commit

Commit the code, `catalog.csv`, and the small `eos_tables/*.dat`. The raw
`eos_raw/` downloads and the `results/` outputs are git-ignored (large /
regenerable).

## Cross-checks (Part 2)

- `compare_to_compose.py` overlays your curve on CompOSE's published `eos.mr`.
- `mr_curve.py` prints the maximum mass and whether it clears the heaviest
  observed pulsar (the EoS-validity test).
- `convergence.py` documents the integrator order (4th-order interior; the full
  star is limited to ~2nd order by the boundary handling).
