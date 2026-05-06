#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <immintrin.h>
#include <sys/time.h>

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

double** alloc_matrix(int n) {
    double* data = (double*)aligned_alloc(32, n * n * sizeof(double));
    double** A = (double**)malloc(n * sizeof(double*));
    if (!data || !A) exit(1);
    for (int i = 0; i < n; ++i) A[i] = data + i * n;
    return A;
}

void free_matrix(double** A) {
    if (A) {
        free(A[0]);
        free(A);
    }
}

void gen_matrix(double** A, double* b, double* x_true, int n) {
    for (int i = 0; i < n; ++i) {
        double row_sum = 0.0;
        for (int j = 0; j < n; ++j) {
            if (i != j) {
                A[i][j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
                row_sum += fabs(A[i][j]);
            }
        }
        A[i][i] = row_sum + 1.0;
    }
    for (int i = 0; i < n; ++i)
        x_true[i] = ((double)rand() / RAND_MAX) * 10.0;
    memset(b, 0, n * sizeof(double));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            b[i] += A[i][j] * x_true[j];
}

double check_residual(double** A, double* b, double* x, int n) {
    double* r = (double*)calloc(n, sizeof(double));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            r[i] += A[i][j] * x[j];
    double norm = 0.0;
    for (int i = 0; i < n; ++i)
        norm += fabs(r[i] - b[i]);
    free(r);
    return norm;
}

void gauss_simd(double** A, double* b, int n, double* x) {
    double* factor = (double*)aligned_alloc(32, n * sizeof(double));
    for (int k = 0; k < n; ++k) {
        double pivot = A[k][k];
        __m256d vpivot = _mm256_set1_pd(pivot);

        int i = k + 1;
        for (; i + 3 < n; i += 4) {
            __m256d aik = _mm256_loadu_pd(&A[i][k]);
            __m256d vfac = _mm256_div_pd(aik, vpivot);
            _mm256_storeu_pd(&factor[i], vfac);
        }
        for (; i < n; ++i) factor[i] = A[i][k] / pivot;

        for (i = k + 1; i < n; ++i) {
            double f = factor[i];
            __m256d vf = _mm256_set1_pd(f);
            int j = k + 1;
            for (; j + 3 < n; j += 4) {
                __m256d a_ij = _mm256_loadu_pd(&A[i][j]);
                __m256d a_kj = _mm256_loadu_pd(&A[k][j]);
                __m256d res = _mm256_fnmadd_pd(vf, a_kj, a_ij);
                _mm256_storeu_pd(&A[i][j], res);
            }
            for (; j < n; ++j) A[i][j] -= f * A[k][j];
            b[i] -= f * b[k];
        }
    }

    for (int i = n-1; i >= 0; --i) {
        __m256d sum_vec = _mm256_setzero_pd();
        int j = i + 1;
        for (; j + 3 < n; j += 4) {
            __m256d a_ij = _mm256_loadu_pd(&A[i][j]);
            __m256d xj = _mm256_loadu_pd(&x[j]);
            sum_vec = _mm256_fmadd_pd(a_ij, xj, sum_vec);
        }
        double sum_arr[4];
        _mm256_storeu_pd(sum_arr, sum_vec);
        double sum = sum_arr[0] + sum_arr[1] + sum_arr[2] + sum_arr[3];
        for (; j < n; ++j) sum += A[i][j] * x[j];
        x[i] = (b[i] - sum) / A[i][i];
    }
    free(factor);
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: %s <matrix_size>\n", argv[0]);
        return 1;
    }
    int n = atoi(argv[1]);
    srand(2024);

    double** A = alloc_matrix(n);
    double** A_bak = alloc_matrix(n);
    double* b = (double*)aligned_alloc(32, n * sizeof(double));
    double* b_bak = (double*)aligned_alloc(32, n * sizeof(double));
    double* x_true = (double*)aligned_alloc(32, n * sizeof(double));
    double* x = (double*)aligned_alloc(32, n * sizeof(double));

    gen_matrix(A, b, x_true, n);
    memcpy(A_bak[0], A[0], n * n * sizeof(double));
    memcpy(b_bak, b, n * sizeof(double));

    double start = get_time();
    gauss_simd(A, b, n, x);
    double end = get_time();
    double elapsed = end - start;

    double residual = check_residual(A_bak, b_bak, x, n);
    printf("n = %d, SIMD time = %.6f s, residual = %.2e\n", n, elapsed, residual);

    free_matrix(A);
    free_matrix(A_bak);
    free(b);
    free(b_bak);
    free(x_true);
    free(x);
    return 0;
}
