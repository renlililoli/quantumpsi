// Minimal KBLAS threading test.
//
// Assumptions:
// - KBLAS math interface is OpenBLAS-style (we call `cblas_dgemm`).
// - KBLAS thread control is aligned with OpenMP (so we control threads via OpenMP).
//
// Build (adjust -lkblas if your library name differs):
//   g++ -O3 -march=native -fopenmp openblas_thread_test.cpp -o kblas_thread_test -lkblas
//
// Run:
//   OMP_NUM_THREADS=1 ./kblas_thread_test 1 900 2
//   OMP_NUM_THREADS=8 ./kblas_thread_test 8 900 2
//
// Args: <omp_threads> <matrix_n> <reps>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>

// For cblas_dgemm we only need the symbol; no vendor headers required.
extern "C" {
    // cblas_dgemm signature:
    // order: 101=row-major, 102=col-major (CBLAS convention)
    void cblas_dgemm(const int order, const int transa, const int transb,
                      const int m, const int n, const int k, const double alpha,
                      const double* A, const int lda, const double* B, const int ldb,
                      const double beta, double* C, const int ldc);
}

#ifdef _OPENMP
#include <omp.h>
#endif

int main(int argc, char** argv) {
    int threads = (argc > 1) ? std::atoi(argv[1]) : 0;
    int n = (argc > 2) ? std::atoi(argv[2]) : 900;
    int reps = (argc > 3) ? std::atoi(argv[3]) : 2;

    // Control threads through OpenMP.
#ifdef _OPENMP
    if (threads > 0) omp_set_num_threads(threads);
    int max_threads = omp_get_max_threads();
    std::cout << "omp_get_max_threads() = " << max_threads << "\n";
    int observed = 1;
#pragma omp parallel
    {
#pragma omp critical
        {
            int nt = omp_get_num_threads();
            if (nt > observed) observed = nt;
        }
    }
    std::cout << "omp parallel region observed_threads = " << observed << "\n";
#else
    std::cout << "Built without OpenMP; omp thread control not available.\n";
#endif

    std::cout << "threads_requested(=omp_threads)=" << threads << " n=" << n << " reps=" << reps << "\n";

    // Force a non-trivial kernel: C = A * B, with A,B full of 1.0
    // Use row-major layout (order=101) and no-transpose (transa/transb=111).
    constexpr int CBLAS_ORDER_ROWMAJOR = 101;
    constexpr int CBLAS_TRANS_N = 111;

    const int m = n, k = n, p = n;
    const double alpha = 1.0;
    const double beta = 0.0;

    std::vector<double> A((size_t)m * k, 1.0);
    std::vector<double> B((size_t)k * p, 1.0);
    std::vector<double> C((size_t)m * p, 0.0);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < reps; r++) {
        cblas_dgemm(CBLAS_ORDER_ROWMAJOR, CBLAS_TRANS_N, CBLAS_TRANS_N,
                    m, p, k, alpha, A.data(), k, B.data(), p, beta, C.data(), p);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();

    double flops = 2.0 * (double)m * (double)p * (double)k; // per GEMM
    double gflops = (flops * reps) / sec / 1e9;

    std::cout << "Elapsed seconds: " << sec << "\n";
    std::cout << "Throughput: " << gflops << " GFLOP/s\n";
    std::cout << "C[0]=" << C[0] << "\n";
    return 0;
}


