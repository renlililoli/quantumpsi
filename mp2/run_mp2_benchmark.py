#!/usr/bin/env python3
"""
DF-MP2 benchmark with Python thin wrapper for basis set loading.

Heavy compute (DF-SCF, DF-MP2 correlation) runs in C++. Same geometry as CCSD.
Use perf to profile: perf record -g -F 99 -- python run_mp2_benchmark.py --threads 8
"""
from __future__ import print_function
import argparse
import sys
import time

# Formic acid dimer (10 atoms) - same as CCSD, suitable for 6226R-level CPU
# From psi4/tests/dfmp2-1
MP2_GEOMETRY = """
0 1
C  -1.888896  -0.179692   0.000000
O  -1.493280   1.073689   0.000000
O  -1.170435  -1.166590   0.000000
H  -2.979488  -0.258829   0.000000
H  -0.498833   1.107195   0.000000
--
0 1
C   1.888896   0.179692   0.000000
O   1.493280  -1.073689   0.000000
O   1.170435   1.166590   0.000000
H   2.979488   0.258829   0.000000
H   0.498833  -1.107195   0.000000
units angstrom
"""


def parse_args():
    p = argparse.ArgumentParser(description="DF-MP2 benchmark (Python basis wrapper)")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--output-file", default="stdout")
    p.add_argument("--csv-file", default="")
    p.add_argument("--geometry", default=None, help="PSI4 geometry string or path to input file")
    return p.parse_args()


def run_one(args):
    import psi4

    psi4.core.clean_timers()
    psi4.core.set_num_threads(args.threads)
    psi4.set_memory("4 GB")

    psi4.set_options({
        "basis": "cc-pvdz",
        "df_basis_scf": "cc-pvdz-jkfit",
        "df_basis_mp2": "cc-pvdz-ri",
        "scf_type": "df",
        "guess": "sad",
        "freeze_core": True,
    })

    if args.geometry:
        try:
            with open(args.geometry) as f:
                geom = f.read()
        except OSError:
            geom = args.geometry
        mol = psi4.geometry(geom)
    else:
        mol = psi4.geometry(MP2_GEOMETRY)

    t0 = time.perf_counter()
    e = psi4.energy("mp2", molecule=mol)
    t1 = time.perf_counter()

    return {
        "energy": e,
        "elapsed_s": t1 - t0,
    }


def main():
    args = parse_args()

    for run in range(1, args.repeat + 1):
        result = run_one(args)

        line = ("run={} threads={} elapsed_s={:.8f} energy={:.16f}".format(
            run, args.threads, result["elapsed_s"], result["energy"]))
        print(line, flush=True)

        if args.csv_file:
            import os
            write_header = run == 1 and not os.path.exists(args.csv_file)
            with open(args.csv_file, "a") as f:
                if write_header:
                    f.write("run,threads,elapsed_s,energy\n")
                f.write("{},{},{:.8f},{:.16f}\n".format(
                    run, args.threads, result["elapsed_s"], result["energy"]))

    try:
        import psi4
        psi4.core.timer_done()
        with open("timer.dat") as t:
            for ln in t:
                if any(x in ln for x in ("DF-MP2", "DF-HF", "MP2")):
                    print("[TIMING] " + ln.rstrip(), flush=True)
    except (IOError, NameError, AttributeError):
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
