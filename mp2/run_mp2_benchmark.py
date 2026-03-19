#!/usr/bin/env python3
"""
DF-MP2 benchmark with Python thin wrapper for basis set loading.

Heavy compute (DF-SCF, DF-MP2 correlation) runs in C++. Same geometry as CCSD.
Use perf to profile: perf record -g -F 99 -- python run_mp2_benchmark.py --threads 8
"""
from __future__ import print_function
import argparse
import os
import sys
import time

# Default: benzene dimer (24 atoms) from benchmark/scf/cases/benzene_dimer.xyz
def _xyz_to_psi4_geom(path):
    """Read XYZ file and return PSI4 geometry string (0 1 + coords + units angstrom)."""
    with open(path) as f:
        n = int(f.readline())
        f.readline()  # skip comment
        lines = [f.readline() for _ in range(n)]
    return "0 1\n" + "".join(lines) + "units angstrom\n"


def parse_args():
    p = argparse.ArgumentParser(description="DF-MP2 benchmark (Python basis wrapper)")
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
        "df_basis_mp2": args.basis.lower() + "-ri",
        "scf_type": "df",
        "guess": "sad",
        "freeze_core": True,
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
        geom_path = os.path.join(os.path.dirname(__file__), "..", "scf", "cases", "benzene_dimer.xyz")
        if not os.path.exists(geom_path):
            raise RuntimeError(
                "No default geometry. Use --geometry <path> or ensure "
                "benchmark/scf/cases/benzene_dimer.xyz exists."
            )
        mol = psi4.geometry(_xyz_to_psi4_geom(geom_path))

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
