"""
run_all.py  --  prepare and run every EoS in the catalog (the Part 3 survey driver).

For each EoS whose raw files are present in eos_raw/<NAME>/:
    1. build the standard table         (build_eos.prepare)
    2. plot the mass-radius curve        (mr_curve.main)
    3. overlay the CompOSE reference      (compare_to_compose.main)
EoS without downloaded raw files are skipped with a note.
"""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import build_eos
import catalog
import mr_curve
import compare_to_compose


def main():
    for row in catalog.load():
        name = row["name"]
        if not os.path.exists(os.path.join("eos_raw", name, "eos.thermo")):
            print(f"\n=== {name}: raw files not found in eos_raw/{name}/, skipping ===")
            continue
        print(f"\n========== {name} ==========")
        build_eos.prepare(name)
        mr_curve.main(name)
        if os.path.exists(os.path.join("eos_raw", name, "eos.mr")):
            compare_to_compose.main(name)


if __name__ == "__main__":
    main()
