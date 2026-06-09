# Neutron Star TOV Solver (SURP 2026)

A from-scratch Tolman–Oppenheimer–Volkoff solver for neutron-star structure,
built for the SURP 2026 project *Neutron Star Rotation and Primordial Black
Hole Accretion*. The goal of Part 2 is to turn a central density into a
neutron-star model `(M, R)`, validate it, and (next) extend it to a central
black hole.

## Files

| file | role |
|------|------|
| `eos.py`         | Equation of state: load APR data from CompOSE, or a toy polytrope for testing |
| `tov.py`         | The structure equations — both Newtonian and relativistic (TOV) right-hand sides |
| `rk4.py`         | One classical 4th-order Runge–Kutta step |
| `solver.py`      | `solve_star(rho_c, eos, ...)` — integrate center → surface, return `(M, R)` |
| `single_star.py` | Build one model and plot its pressure/mass profiles |
| `mr_curve.py`    | Sweep central density → mass–radius relation (with the Newtonian curve for contrast) |
| `convergence.py` | Verify the order of the scheme |

## Getting the EOS data

Download the **APR** EOS from CompOSE (https://compose.obspm.fr/eos/297) and
put `eos.nb` and `eos.thermo` in this folder. Column convention (CompOSE
manual p.37): `Q1 = p/n_b` is thermo column index 3, `Q7 = e/(n_b m_n c^2) − 1`
is thermo column index 9. The mass-energy density is `rho = (Q7 + 1) n_b m_n`.

Without the data files, every script falls back to a polytrope so you can still
test the code.

## Run

```bash
python single_star.py     # one model + profile plot
python mr_curve.py         # full mass-radius curve
python convergence.py      # order-of-convergence check
```

## Cross-checks (Part 2)

1. A single model should land on one point of the published APR M–R curve.
2. Sweeping `rho_c` should reproduce the whole published curve.
3. The peak of the curve is the maximum mass — it must exceed the observed
   most-massive neutron star for the EOS to be allowed.
4. Halving the step size should shrink the error (see `convergence.py` for the
   subtlety about which parts of the scheme are 4th vs 2nd order).
   
## Roadmap

- [x] TOV solver + single star
- [x] mass–radius curve + Newtonian comparison
- [x] convergence check
- [ ] reproduce published APR curve with real CompOSE data
- [ ] ATNF pulsar catalog: rotation-period distribution
- [ ] PBH extension: inner boundary at `R_BH`, sweep `(M_BH, rho_c)`
- [ ] 2D `(M, R)` survey with constant-mass accretion tracks
- [ ] Kepler-velocity ejection condition
