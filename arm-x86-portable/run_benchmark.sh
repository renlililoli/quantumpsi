#!/bin/bash
#
# ARM/x86 多核 Benchmark 采集脚本（跨平台便携版）
#
# 依赖：bash、Python 3、已配置好的 Psi4 环境。不依赖 MKL，支持 OpenBLAS 等。
#
# 用法:
#   在 x86 上: ./run_benchmark.sh --platform x86 [--project-root /path/to/quantum-psi]
#   在 ARM 上: ./run_benchmark.sh --platform arm [--project-root /path/to/quantum-psi]
#
# 参数:
#   --platform arm|x86    必填
#   --project-root DIR    quantum-psi 项目根目录，含 benchmark/scf 等
#   --threads "1 2 4 8"  线程点
#   --repeat 5           每点重复次数
#   --methods "scf dft mp2 sapt0 ccsd gradient hessian frequency"
#   --results-dir DIR    结果目录，默认 <project-root>/results
#
set -e

PORTABLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT=""
RESULTS_DIR=""
PLATFORM=""
THREADS="1 2 4 8 16 32"
REPEAT=5
METHODS="scf dft mp2 sapt0 ccsd gradient hessian frequency"
USE_TASKSET=1

# Detect one logical CPU per physical core.
# Output: space-separated cpu ids, e.g. "0 1 2 ...".
detect_physical_cores() {
    local cores
    cores=$(lscpu -p=CPU,CORE,SOCKET 2>/dev/null | awk -F, '
        BEGIN { OFS="," }
        /^#/ { next }
        {
            key = $2 ":" $3
            if (!(key in seen)) {
                seen[key] = 1
                print $1
            }
        }' | sort -n | tr '\n' ' ')
    echo "$cores"
}

# Build a taskset cpulist for t threads from physical cores first.
build_cpuset_for_threads() {
    local t=$1
    local cores=($2)
    local n=${#cores[@]}
    local cpuset=""

    if [[ $n -eq 0 ]]; then
        echo ""
        return 1
    fi

    if (( t > n )); then
        echo "WARN: requested threads=$t > physical_cores=$n, taskset will use first $n physical cores." >&2
        t=$n
    fi

    local i
    for ((i=0; i<t; i++)); do
        if [[ -z "$cpuset" ]]; then
            cpuset="${cores[$i]}"
        else
            cpuset="${cpuset},${cores[$i]}"
        fi
    done
    echo "$cpuset"
    return 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --platform)      PLATFORM="$2"; shift 2 ;;
        --project-root) PROJECT_ROOT="$2"; shift 2 ;;
        --threads)      THREADS="$2"; shift 2 ;;
        --repeat)       REPEAT="$2";  shift 2 ;;
        --methods)      METHODS="$2"; shift 2 ;;
        --results-dir)  RESULTS_DIR="$2"; shift 2 ;;
        --no-taskset)   USE_TASKSET=0; shift 1 ;;
        -h|--help)
            echo "Usage: $0 --platform arm|x86 [--project-root DIR] [--threads \"1 2 4 8\"] [--repeat 5] [--no-taskset]"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$PLATFORM" ]]; then
    echo "ERROR: --platform arm or x86 is required"
    exit 1
fi

# 默认 project root：便携包所在目录的上两级 (benchmark/arm-x86-portable -> .)
if [[ -z "$PROJECT_ROOT" ]]; then
    PROJECT_ROOT="$(cd "$PORTABLE_DIR/../.." && pwd)"
fi
PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"

if [[ -z "$RESULTS_DIR" ]]; then
    RESULTS_DIR="$PROJECT_ROOT/results"
fi

# 校验 benchmark 脚本存在
if [[ ! -f "$PROJECT_ROOT/benchmark/scf/run_scf_benchmark.py" ]]; then
    echo "ERROR: benchmark scripts not found in $PROJECT_ROOT"
    echo "Specify --project-root /path/to/quantum-psi"
    exit 1
fi

cd "$PROJECT_ROOT"

# # 加载环境（若有）
# if [[ -f "$PROJECT_ROOT/workspace/.env" ]]; then
#     source "$PROJECT_ROOT/workspace/.env"
# fi

OUT_DIR="$RESULTS_DIR/$PLATFORM"
mkdir -p "$OUT_DIR"
RAW_CSV="$OUT_DIR/benchmark_raw.csv"

echo "platform,method,threads,run_idx,elapsed_s,energy" > "$RAW_CSV"

PHYSICAL_CORES="$(detect_physical_cores)"
if [[ -z "$PHYSICAL_CORES" ]]; then
    echo "WARN: failed to detect physical cores via lscpu; taskset binding disabled." >&2
    USE_TASKSET=0
fi

if [[ "$USE_TASKSET" -eq 1 ]]; then
    echo "taskset binding enabled (one thread per physical core)" >&2
fi

run_benchmark() {
    local method=$1 t=$2 run_idx=$3
    local elapsed energy out err cpuset pycmd
    err=$(mktemp)
    cpuset=""
    pycmd="python"

    if [[ "$USE_TASKSET" -eq 1 ]]; then
        cpuset=$(build_cpuset_for_threads "$t" "$PHYSICAL_CORES" || true)
        if [[ -n "$cpuset" ]]; then
            pycmd="taskset -c $cpuset python"
        fi
    fi

    case $method in
        scf)
            out=$($pycmd benchmark/scf/run_scf_benchmark.py --threads "$t" --repeat 1 2>"$err" | grep "^run=" || true)
            ;;
        dft)
            out=$($pycmd benchmark/dft/run_dft_benchmark.py --threads "$t" --repeat 1 2>"$err" | grep "^run=" || true)
            ;;
        mp2)
            out=$($pycmd benchmark/mp2/run_mp2_benchmark.py --threads "$t" --repeat 1 2>"$err" | grep "^run=" || true)
            ;;
        sapt0)
            out=$($pycmd benchmark/sapt0/run_sapt0_benchmark.py --threads "$t" --repeat 1 2>"$err" | grep "^run=" || true)
            ;;
        ccsd)
            out=$($pycmd benchmark/ccsd/run_ccsd_benchmark.py --threads "$t" --repeat 1 2>"$err" | grep "^run=" || true)
            ;;
        gradient)
            out=$($pycmd benchmark/gradient/run_gradient_benchmark.py --threads "$t" --repeat 1 2>"$err" | grep "^run=" || true)
            ;;
        hessian)
            out=$($pycmd benchmark/hessian/run_hessian_benchmark.py --threads "$t" --repeat 1 2>"$err" | grep "^run=" || true)
            ;;
        frequency)
            out=$($pycmd benchmark/frequency/run_frequency_benchmark.py --threads "$t" --repeat 1 2>"$err" | grep "^run=" || true)
            ;;
        *) echo "Unknown method: $method"; return 1 ;;
    esac
    elapsed=$(echo "$out" | sed -n 's/.*elapsed_s=\([0-9.]*\).*/\1/p')
    energy=$(echo "$out"  | sed -n 's/.*energy=\(-\?[0-9.]*\).*/\1/p')
    if [[ -z "$elapsed" ]]; then
        echo "WARN: Failed to parse elapsed_s for $method t=$t run=$run_idx" >&2
        [[ -s "$err" ]] && tail -20 "$err" >&2
        rm -f "$err"
        return 1
    fi
    rm -f "$err"
    echo "${PLATFORM},${method},${t},${run_idx},${elapsed},${energy}" >> "$RAW_CSV"
}

echo "=========================================="
echo "ARM/x86 Benchmark: platform=$PLATFORM project=$PROJECT_ROOT"
echo "Threads=$THREADS repeat=$REPEAT results=$OUT_DIR"
echo "=========================================="

for method in $METHODS; do
    for t in $THREADS; do
        echo "[$method] threads=$t ..."
        for i in $(seq 1 $REPEAT); do
            run_benchmark "$method" "$t" "$i" || true
        done
    done
done

{
    echo "platform=$PLATFORM"
    echo "date=$(date -Iseconds 2>/dev/null || date)"
    uname -a
    echo "---"
    (lscpu 2>/dev/null || true)
    echo "---"
    (numactl --hardware 2>/dev/null || true)
} > "$OUT_DIR/platform_info.txt"

echo ""
echo "Done. Raw data: $RAW_CSV"
echo "Generate report: python $PORTABLE_DIR/generate_report.py --results-dir $RESULTS_DIR"
