#!/usr/bin/env python3
"""
DFT (B3LYP) benchmark with Python thin wrapper for basis set loading.

Heavy compute (xc integration, LibXC, DF-JK) runs in C++. Same geometry as SCF.
Use perf to profile: perf record -g -F 99 -- python run_dft_benchmark.py --threads 8 --single-iter
"""
from __future__ import print_function
import argparse
import os
import sys
import time

# Default: benzene dimer cc-pVTZ, heavier for multicore scaling
def _xyz_to_psi4_geom(path):
    """Read XYZ file and return PSI4 geometry string (0 1 + coords + units angstrom)."""
    with open(path) as f:
        n = int(f.readline())
        f.readline()  # skip comment
        lines = [f.readline() for _ in range(n)]
    return "0 1\n" + "".join(lines) + "units angstrom\n"


def parse_args():
    p = argparse.ArgumentParser(description="DFT (B3LYP) benchmark (Python basis wrapper)")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--single-iter", action="store_true",
                   help="Run SCF with maxiter=1 (one iteration) for perf focus")
    p.add_argument("--basis", default="cc-pVTZ", help="Orbital basis set (default: cc-pVTZ)")
    p.add_argument("--output-file", default="stdout")
    p.add_argument("--csv-file", default="")
    p.add_argument("--geometry", default=None, help="PSI4 geometry string or path to XYZ file")
    return p.parse_args()


def run_one(args):
    import psi4

    psi4.core.clean_timers()
    psi4.core.set_num_threads(args.threads)
    psi4.set_memory("4 GB")

    opts = {
        "basis": args.basis,
        "scf_type": "df",
        "df_basis_scf": args.basis.lower() + "-jkfit",
        "e_convergence": 1e-8,
        "d_convergence": 1e-8,
        "reference": "rhf",
    }
    if args.single_iter:
        opts["maxiter"] = 1
    psi4.set_options(opts)

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
        geom_path = os.path.join(os.path.dirname(__file__), "..", "scf", "cases", "benzene_dimer.xyz")
        if not os.path.exists(geom_path):
            raise RuntimeError(
                "No default geometry. Use --geometry <path> or ensure "
                "benchmark/scf/cases/benzene_dimer.xyz exists."
            )
        mol = psi4.geometry(_xyz_to_psi4_geom(geom_path))

    t0 = time.perf_counter()
    e, wfn = psi4.energy("b3lyp", molecule=mol, return_wfn=True)
    t1 = time.perf_counter()

    nbf = wfn.basisset().nbf() if wfn else 0
    return {
        "energy": e,
        "elapsed_s": t1 - t0,
        "nbf": nbf,
    }


def main():
    args = parse_args()

    for run in range(1, args.repeat + 1):
        result = run_one(args)

        line = ("run={} threads={} elapsed_s={:.8f} energy={:.16f}".format(
            run, args.threads, result["elapsed_s"], result["energy"]))
        print(line, flush=True)

        if args.csv_file:
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
                if any(x in ln for x in ("V V", "DF-JK", "xc ", "LibXC")):
                    print("[TIMING] " + ln.rstrip(), flush=True)
    except (IOError, NameError, AttributeError):
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
