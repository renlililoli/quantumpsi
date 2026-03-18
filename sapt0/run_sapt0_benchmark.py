#!/usr/bin/env python3
"""
SAPT0 benchmark with Python thin wrapper for basis set loading.

Heavy compute (dimer SCF, monomer SCF, SAPT0::compute_energy) runs in C++.
Use perf to profile: perf record -g -F 99 -- python run_sapt0_benchmark.py --threads 8
"""
from __future__ import print_function
import argparse
import sys
import time

# Formamide dimer (24 atoms, aug-cc-pVDZ) - suitable for 6226R-level CPU
# From psi4/tests/sapt4
SAPT0_GEOMETRY = """
0 1
C  -2.018649   0.052883   0.000000
O  -1.452200   1.143634   0.000000
N  -1.407770  -1.142484   0.000000
H  -1.964596  -1.977036   0.000000
H  -0.387244  -1.207782   0.000000
H  -3.117061  -0.013701   0.000000
--
0 1
C   2.018649  -0.052883   0.000000
O   1.452200  -1.143634   0.000000
N   1.407770   1.142484   0.000000
H   1.964596   1.977036   0.000000
H   0.387244   1.207782   0.000000
H   3.117061   0.013701   0.000000
units angstrom
"""


def parse_args():
    p = argparse.ArgumentParser(description="SAPT0 benchmark (Python basis wrapper)")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--output-file", default="stdout")
    p.add_argument("--csv-file", default="")
    p.add_argument("--geometry", default=None, help="PSI4 geometry string or path to input file")
    return p.parse_args()


def run_one(args):
    import psi4

    psi4.core.clean_timers()  # Reset so timer.dat reflects this run only
    psi4.core.set_num_threads(args.threads)
    psi4.set_memory("4 GB")

    psi4.set_options({
        "basis": "aug-cc-pVDZ",
        "scf_type": "df",
        "freeze_core": True,
        "df_basis_sapt": "aug-cc-pVDZ-RI",
        "df_basis_scf": "aug-cc-pVDZ-JKFIT",
        "e_convergence": 1e-8,
        "d_convergence": 1e-8,
    })

    if args.geometry:
        try:
            with open(args.geometry) as f:
                geom = f.read()
        except OSError:
            geom = args.geometry
        dimer = psi4.geometry(geom)
    else:
        dimer = psi4.geometry(SAPT0_GEOMETRY)

    t0 = time.perf_counter()
    e = psi4.energy("sapt0", molecule=dimer)
    t1 = time.perf_counter()

    return {
        "energy": e,
        "elapsed_s": t1 - t0,
        "nbf": dimer.natom() * 10,  # approx; can get from wfn if needed
    }


def main():
    args = parse_args()

    for run in range(1, args.repeat + 1):
        result = run_one(args)

        line = ("run={} threads={} elapsed_s={:.8f} energy={:.16f}".format(
            run, args.threads, result["elapsed_s"], result["energy"]))
        print(line, flush=True)

        if args.csv_file:
            write_header = run == 1 and not __import__("os").path.exists(args.csv_file)
            with open(args.csv_file, "a") as f:
                if write_header:
                    f.write("run,threads,elapsed_s,energy\n")
                f.write("{},{},{:.8f},{:.16f}\n".format(
                    run, args.threads, result["elapsed_s"], result["energy"]))

    # Flush C++ timers and print SAPT phase breakdown (only after last run)
    try:
        import psi4
        psi4.core.timer_done()
        with open("timer.dat") as t:
            for ln in t:
                if any(x in ln for x in ("SAPT", "DF Integrals", "Elst", "Exch", "Ind", "Disp", "W Integrals")):
                    print("[TIMING] " + ln.rstrip(), flush=True)
    except (IOError, NameError, AttributeError):
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
