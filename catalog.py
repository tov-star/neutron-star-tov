"""
catalog.py  --  read the EoS registry (catalog.csv).

Each row records one equation of state and its metadata:
    name        short name (used for folder/table/results names)
    compose_id  CompOSE database id (provenance -- so you remember what it is)
    model       short description of the physical model
    dim         "1D" or "3D" (informational; build_eos.py auto-detects anyway)
    ref_Mmax    maximum mass from the CompOSE page [Msun]  (for validation)
    ref_R14     radius at 1.4 Msun from the CompOSE page [km]
"""

import csv
import os

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "catalog.csv")

# Heaviest well-measured pulsar -- the project's EoS-validity threshold.
OBS_MAX_MASS = 2.08    # Msun (PSR J0740+6620, approx)


def load(path=PATH):
    with open(path) as f:
        return list(csv.DictReader(f))


def get(name, path=PATH):
    for row in load(path):
        if row["name"] == name:
            return row
    raise KeyError(f"'{name}' is not in {path}")
