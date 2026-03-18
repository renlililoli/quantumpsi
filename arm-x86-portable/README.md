# ARM/x86 多核对比工具（跨平台便携包）

本目录为**不依赖数学库（MKL/OpenBLAS）和重型 Python 包**的便携工具，可单独打包到任意机器使用。

## 依赖说明

| 文件 | 依赖 |
|------|------|
| `generate_report.py` | 仅 Python 3 标准库（无 numpy、无 BLAS） |
| `run_benchmark.sh` | bash，运行时需调用量子项目中的 Psi4 benchmark |
| `README.md` | 无 |

## 打包方式

```bash
# 只打包本目录
tar -czvf arm-x86-portable.tar.gz benchmark/arm-x86-portable/

# 或 zip
zip -r arm-x86-portable.zip benchmark/arm-x86-portable/
```

拷贝到目标机器解压即可使用。

## 使用流程

### 1. 在 x86 机器上采集

在 quantum-psi 项目根目录下：

```bash
source workspace/.env   # 或按需配置 Psi4 环境
./benchmark/arm-x86-portable/run_benchmark.sh --platform x86
```

若便携包不在项目内，指定项目路径：

```bash
./run_benchmark.sh --platform x86 --project-root /path/to/quantum-psi
```

结果保存到 `results/x86/benchmark_raw.csv`。

### 2. 在 ARM 机器上采集

```bash
./run_benchmark.sh --platform arm --project-root /path/to/quantum-psi
```

ARM 核多时可加线程点：

```bash
./run_benchmark.sh --platform arm --project-root /path/to/quantum-psi \
  --threads "1 2 4 8 16 32 48 64 96"
```

### 3. 生成报告

将 `results/arm/` 和 `results/x86/` 放到同一目录，在**任意机器**上运行（只需 Python 3）：

```bash
python generate_report.py --results-dir /path/to/results
```

报告输出到 `results/reports/arm_x86_report_YYYY-MM-DD_HHMM.md`。

## 报告生成器 standalone 用法

若仅需生成报告（已有 CSV 数据），可将 `generate_report.py` 单独拷贝，无需项目其余部分：

```bash
python generate_report.py --results-dir /path/to/results -o report.md
```

`results/` 目录结构：

```
results/
  arm/
    benchmark_raw.csv
  x86/
    benchmark_raw.csv
```

## 参数一览

**run_benchmark.sh**

| 参数 | 默认 | 说明 |
|------|------|------|
| `--platform` | 必填 | arm / x86 |
| `--project-root` | 便携包上两级 | quantum-psi 项目根 |
| `--threads` | 1 2 4 8 16 32 | 线程点 |
| `--repeat` | 5 | 每点重复次数 |
| `--methods` | scf sapt0 ccsd | benchmark 类型 |
| `--results-dir` | &lt;project&gt;/results | 结果根目录 |

**generate_report.py**

| 参数 | 默认 | 说明 |
|------|------|------|
| `--results-dir` | . | 含 arm/、x86/ 的目录 |
| `-o` | auto | 报告输出路径 |
