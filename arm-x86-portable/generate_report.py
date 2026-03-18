#!/usr/bin/env python3
"""
ARM/x86 多核对比报告生成器（跨平台便携版）

依赖：仅 Python 3 标准库，无 numpy、无 BLAS/MKL/OpenBLAS。
可在任意机器上运行，只需 results/arm/benchmark_raw.csv 和 results/x86/benchmark_raw.csv。

用法: python generate_report.py [--results-dir DIR] [-o output.md]
"""
from __future__ import print_function
import argparse
import os
import statistics
from datetime import datetime


def load_raw_csv(path):
    """Load raw CSV, return list of dicts."""
    rows = []
    with open(path) as f:
        header = f.readline().strip().split(",")
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            row = dict(zip(header, parts))
            try:
                row["threads"] = int(row["threads"])
                row["run_idx"] = int(row["run_idx"])
                row["elapsed_s"] = float(row["elapsed_s"])
                rows.append(row)
            except (ValueError, KeyError):
                continue
    return rows


def agg_by_platform_method_threads(rows):
    """Aggregate: median, stdev of elapsed_s per (platform, method, threads)."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        key = (r["platform"], r["method"], r["threads"])
        groups[key].append(r["elapsed_s"])

    result = {}
    for (plat, method, t), vals in groups.items():
        result[(plat, method, t)] = {
            "median_s": float(statistics.median(vals)),
            "stdev_s": float(statistics.stdev(vals)) if len(vals) > 1 else 0.0,
            "energy": None,
        }
    return result


def compute_speedup_efficiency(agg):
    """For each (platform, method), compute speedup and efficiency vs T1."""
    result = {}
    for (plat, method, t), v in agg.items():
        t1_key = (plat, method, 1)
        if t1_key not in agg:
            continue
        t1 = agg[t1_key]["median_s"]
        tn = v["median_s"]
        speedup = t1 / tn if tn > 0 else 0
        efficiency = speedup / t if t > 0 else 0
        result[(plat, method, t)] = {
            **v,
            "speedup": speedup,
            "efficiency": efficiency,
        }
    return result


def arm_cores_to_match_x86(agg, x86_baseline_threads=8):
    """For each method: how many ARM cores to match x86 at x86_baseline_threads?"""
    out = {}
    for method in ["scf", "dft", "mp2", "sapt0", "ccsd", "gradient", "hessian", "frequency"]:
        x86_threads = sorted(set(k[2] for k in agg if k[0] == "x86" and k[1] == method))
        if not x86_threads:
            continue
        target_t = min(x86_threads, key=lambda x: abs(x - x86_baseline_threads))
        target_time = agg.get(("x86", method, target_t), {}).get("median_s")
        if target_time is None:
            continue

        arm_keys = [(k[2], agg[k]["median_s"]) for k in agg
                    if k[0] == "arm" and k[1] == method]
        arm_times = sorted(arm_keys, key=lambda x: x[0])
        if not arm_times:
            continue

        best_arm_cores = None
        for t, s in arm_times:
            if s <= target_time:
                best_arm_cores = t
                break
        if best_arm_cores is None:
            best_arm_cores = arm_times[-1][0]
        out[method] = (best_arm_cores, target_time)
    return out


def build_report(agg, speedup_data, arm_match, results_dir):
    """Build markdown report."""
    lines = []
    lines.append("# ARM vs x86 多核性能对比报告")
    lines.append("")
    lines.append(f"生成时间: {datetime.now().isoformat()}")
    lines.append("")

    lines.append("## 1. 对比指标说明")
    lines.append("")
    lines.append("| 指标 | 含义 |")
    lines.append("|------|------|")
    lines.append("| median_s | 单次运行墙钟时间中位数 (秒) |")
    lines.append("| speedup | 相对 1 核加速比 = T1 / Tn |")
    lines.append("| efficiency | 并行效率 = speedup / n |")
    lines.append("| ARM 追平 x86 所需核数 | ARM 需多少核才能达到 x86 在 8 核下的耗时 |")
    lines.append("")

    lines.append("## 2. 原始数据汇总表")
    lines.append("")
    lines.append("| platform | method | threads | median_s | stdev_s | speedup | efficiency |")
    lines.append("|----------|--------|---------|----------|---------|---------|------------|")

    for plat in ["x86", "arm"]:
        for method in ["scf", "dft", "mp2", "sapt0", "ccsd", "gradient", "hessian", "frequency"]:
            for t in sorted(set(k[2] for k in speedup_data if k[0] == plat and k[1] == method)):
                key = (plat, method, t)
                if key not in speedup_data:
                    continue
                v = speedup_data[key]
                lines.append("| {} | {} | {} | {:.3f} | {:.3f} | {:.2f} | {:.1%} |".format(
                    plat, method, t, v["median_s"], v["stdev_s"],
                    v.get("speedup", 0), v.get("efficiency", 0)))

    lines.append("")
    lines.append("## 3. ARM 追平 x86 所需核心数 (x86 基线: 8 核)")
    lines.append("")
    lines.append("| method | x86 8核耗时 (s) | ARM 追平所需核数 |")
    lines.append("|--------|----------------|------------------|")
    for method, (arm_cores, x86_time) in arm_match.items():
        lines.append("| {} | {:.3f} | {} |".format(method, x86_time, arm_cores))
    lines.append("")

    lines.append("## 4. 结论要点")
    lines.append("")
    lines.append("1. **同核数对比**：在 1/2/4/8 核下，ARM 与 x86 的 wall time 比值。")
    lines.append("2. **扩展性**：ARM 若核数更多，能否通过堆核追平 x86 的 8 核性能。")
    lines.append("3. **效率塌陷**：若 ARM 需 32+ 核才追平 x86 8 核，且 efficiency < 35%，则单位核效弱于 x86。")
    lines.append("")

    lines.append("## 5. 原始数据文件")
    lines.append("")
    for plat in ["x86", "arm"]:
        p = os.path.join(results_dir, plat, "benchmark_raw.csv")
        if os.path.exists(p):
            lines.append("- `{}/benchmark_raw.csv`".format(plat))
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Generate ARM/x86 benchmark report (portable, Python stdlib only)")
    ap.add_argument("--results-dir", default=".", help="Directory containing arm/ and x86/ subdirs with benchmark_raw.csv")
    ap.add_argument("-o", "--output", default=None, help="Output report path")
    args = ap.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    rows = []

    for plat in ["x86", "arm"]:
        csv_path = os.path.join(results_dir, plat, "benchmark_raw.csv")
        if not os.path.exists(csv_path):
            print("Warning: {} not found, skipping".format(csv_path))
            continue
        rows.extend(load_raw_csv(csv_path))

    if not rows:
        print("No data found. Run run_benchmark.sh on both platforms first.")
        return 1

    agg = agg_by_platform_method_threads(rows)
    speedup_data = compute_speedup_efficiency(agg)
    arm_match = arm_cores_to_match_x86(agg, x86_baseline_threads=8)

    report = build_report(agg, speedup_data, arm_match, results_dir)

    out_path = args.output
    if not out_path:
        reports_dir = os.path.join(results_dir, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        out_path = os.path.join(reports_dir, "arm_x86_report_{}.md".format(
            datetime.now().strftime("%Y-%m-%d_%H%M")))

    with open(out_path, "w") as f:
        f.write(report)

    print("Report written to:", out_path)
    return 0


if __name__ == "__main__":
    exit(main())
