// Minimal OpenMP-controlled BLAS threading test.
//
// Assumptions:
// - BLAS provides OpenBLAS-style CBLAS symbol: cblas_dgemm
// - The BLAS library is built with OpenMP backend, so its internal threading
//   is controlled by the OpenMP runtime.
//
// Build:
//   g++ -O3 -march=native -fopenmp openmp_blas_thread_test.cpp -o openmp_blas_thread_test -lkblas
//
// Run examples:
//   OMP_NUM_THREADS=1 ./openmp_blas_thread_test 900 2
//   OMP_NUM_THREADS=8 ./openmp_blas_thread_test 900 2
//
// Args: <matrix_n> <reps>
//
// Notes:
// - Thread count is controlled ONLY by OpenMP.
// - Do NOT wrap cblas_dgemm in an outer omp parallel region unless you want nested parallelism.

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>

extern "C" {
    void cblas_dgemm(const int order, const int transa, const int transb,
                     const int m, const int n, const int k, const double alpha,
                     const double* A, const int lda, const double* B, const int ldb,
                     const double beta, double* C, const int ldc);
}

#ifdef _OPENMP
#include <omp.h>
#endif

int main(int argc, char** argv) {
    int n    = (argc > 1) ? std::atoi(argv[1]) : 900;
    int reps = (argc > 2) ? std::atoi(argv[2]) : 2;

#ifdef _OPENMP
    // 完全交给 OpenMP 控制
    // 1) 禁止动态调整线程数，避免运行时偷偷改线程数
    omp_set_dynamic(0);

    // 2) 默认关闭嵌套并行，避免外层并行 + BLAS 内层并行导致过度订阅
    omp_set_max_active_levels(1);

    // 仅用于打印诊断信息
    std::cout << "OpenMP enabled\n";
    std::cout << "omp_get_max_threads() = " << omp_get_max_threads() << "\n";

    int observed = 1;
#pragma omp parallel
    {
#pragma omp single
        {
            observed = omp_get_num_threads();
        }
    }
    std::cout << "omp parallel region observed_threads = " << observed << "\n";
#else
    std::cout << "Built without OpenMP; BLAS threading cannot be controlled by OpenMP.\n";
#endif

    std::cout << "n=" << n << " reps=" << reps << "\n";

    constexpr int CBLAS_ORDER_ROWMAJOR = 101;
    constexpr int CBLAS_TRANS_N = 111;

    const int m = n, k = n, p = n;
    const double alpha = 1.0;
    const double beta  = 0.0;

    std::vector<double> A((size_t)m * k, 1.0);
    std::vector<double> B((size_t)k * p, 1.0);
    std::vector<double> C((size_t)m * p, 0.0);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < reps; ++r) {
        cblas_dgemm(CBLAS_ORDER_ROWMAJOR, CBLAS_TRANS_N, CBLAS_TRANS_N,
                    m, p, k,
                    alpha,
                    A.data(), k,
                    B.data(), p,
                    beta,
                    C.data(), p);
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    double sec = std::chrono::duration<double>(t1 - t0).count();
    double flops = 2.0 * (double)m * (double)p * (double)k;
    double gflops = (flops * reps) / sec / 1e9;

    std::cout << "Elapsed seconds: " << sec << "\n";
    std::cout << "Throughput: " << gflops << " GFLOP/s\n";
    std::cout << "C[0] = " << C[0] << "\n";

    return 0;
}
