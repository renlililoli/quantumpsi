#!/usr/bin/env python3
"""
HF analytic gradient benchmark with Python thin wrapper.

Heavy compute (DF-SCF, gradient) runs in C++. Same geometry as MP2/CCSD.
Use perf to profile: perf record -g -F 99 -- python run_gradient_benchmark.py --threads 8
"""
from __future__ import print_function
import argparse
import sys
import time

# Benzene monomer (12 atoms), extracted from benzene_dimer.xyz first monomer
GRADIENT_GEOMETRY = """
0 1
C   1.391500   0.000000   0.000000
C   0.695750   1.205074   0.000000
C  -0.695750   1.205074   0.000000
C  -1.391500   0.000000   0.000000
C  -0.695750  -1.205074   0.000000
C   0.695750  -1.205074   0.000000
H   2.471500   0.000000   0.000000
H   1.235750   2.140382   0.000000
H  -1.235750   2.140382   0.000000
H  -2.471500   0.000000   0.000000
H  -1.235750  -2.140382   0.000000
H   1.235750  -2.140382   0.000000
units angstrom
"""


def parse_args():
    p = argparse.ArgumentParser(description="HF gradient benchmark (analytic, Python basis wrapper)")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--basis", default="cc-pvtz", help="Orbital basis set (default: cc-pvtz)")
    p.add_argument("--output-file", default="stdout")
    p.add_argument("--csv-file", default="")
    p.add_argument("--geometry", default=None, help="PSI4 geometry string or path to XYZ/input file")
    return p.parse_args()


def run_one(args):
    import psi4

    psi4.core.clean_timers()
    psi4.core.set_num_threads(args.threads)
    psi4.set_memory("4 GB")

    psi4.set_options({
        "basis": args.basis,
        "df_basis_scf": args.basis.lower() + "-jkfit",
        "scf_type": "df",
        "guess": "sad",
    })

    if args.geometry:
        try:
            with open(args.geometry) as f:
                first = f.readline()
                try:
                    n = int(first)
                    f.readline()  # skip comment/charge line
                    lines = [f.readline() for _ in range(n)]
                    geom = "0 1\n" + "".join(lines) + "units angstrom\n"
                except ValueError:
                    geom = first + f.read()
        except OSError:
            geom = args.geometry
        mol = psi4.geometry(geom)
    else:
        mol = psi4.geometry(GRADIENT_GEOMETRY)

    t0 = time.perf_counter()
    grad, wfn = psi4.gradient("hf/{}".format(args.basis.lower()), molecule=mol, dertype=1, return_wfn=True)
    t1 = time.perf_counter()

    e = wfn.energy() if wfn else 0.0
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
                if any(x in ln for x in ("Gradient", "DF-HF", "JK")):
                    print("[TIMING] " + ln.rstrip(), flush=True)
    except (IOError, NameError, AttributeError):
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
