#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <immintrin.h>
#include <pthread.h>
#include <omp.h>

// 分配二维矩阵（基于你原有的连续内存+行指针结构）
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
        A[i][i] = row_sum + 2.0; // 严格对角占优
        x_true[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    }
    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        for (int j = 0; j < n; ++j) sum += A[i][j] * x_true[j];
        b[i] = sum;
    }
}

// 校验正确性
int check_solution(double* x, double* x_true, int n) {
    for (int i = 0; i < n; ++i) {
        if (fabs(x[i] - x_true[i]) > 1e-4) return 0;
    }
    return 1;
}

// ==========================================
// 💡 补全核心：双精度 SIMD 优化串行回代函数
// ==========================================
void back_substitution(double** A, double* b, int n, double* x) {
    for (int i = n - 1; i >= 0; --i) {
        __m256d sum_vec = _mm256_setzero_pd();
        int j = i + 1;
        
        // 4路双精度浮点 FMA 乘加加速
        for (; j + 3 < n; j += 4) {
            __m256d v_a = _mm256_loadu_pd(&A[i][j]);
            __m256d v_x = _mm256_loadu_pd(&x[j]);
            sum_vec = _mm256_fmadd_pd(v_a, v_x, sum_vec);
        }
        
        double sum_arr[4];
        _mm256_storeu_pd(sum_arr, sum_vec);
        double sum = sum_arr[0] + sum_arr[1] + sum_arr[2] + sum_arr[3];
        
        // 补齐边界
        for (; j < n; ++j) {
            sum += A[i][j] * x[j];
        }
        x[i] = (b[i] - sum) / A[i][i];
    }
}

// ==========================================
// 1. 基本要求：基础 Pthread + SIMD 结合版
// ==========================================
typedef struct {
    int t_id;
    int t_count;
    int k;
    int n;
    double** A;
    double* b;
} pthread_args_t;

void* pthread_elim_worker(void* arg) {
    pthread_args_t* args = (pthread_args_t*)arg;
    int k = args->k;
    int n = args->n;
    double** A = args->A;
    double* b = args->b;

    for (int i = k + 1 + args->t_id; i < n; i += args->t_count) {
        double factor = A[i][k] / A[k][k];
        
        int j = k + 1;
        __m256d v_factor = _mm256_set1_pd(factor);
        for (; j + 3 < n; j += 4) {
            __m256d v_ak = _mm256_loadu_pd(&A[k][j]);
            __m256d v_ai = _mm256_loadu_pd(&A[i][j]);
            __m256d v_res = _mm256_sub_pd(v_ai, _mm256_mul_pd(v_factor, v_ak));
            _mm256_storeu_pd(&A[i][j], v_res);
        }
        for (; j < n; ++j) {
            A[i][j] -= factor * A[k][j];
        }
        b[i] -= factor * b[k];
    }
    pthread_exit(NULL);
}

void gauss_pthread_simd(double** A, double* b, int n, int threads) {
    for (int k = 0; k < n; ++k) {
        pthread_t* handles = malloc(sizeof(pthread_t) * threads);
        pthread_args_t* args = malloc(sizeof(pthread_args_t) * threads);

        for (int t = 0; t < threads; t++) {
            args[t].t_id = t;
            args[t].t_count = threads;
            args[t].k = k;
            args[t].n = n;
            args[t].A = A;
            args[t].b = b;
            pthread_create(&handles[t], NULL, pthread_elim_worker, (void*)&args[t]);
        }
        for (int t = 0; t < threads; t++) {
            pthread_join(handles[t], NULL);
        }
        free(handles); free(args);
    }
}

// ==========================================
// 2. 基本要求：基础 OpenMP + SIMD 结合版
// ==========================================
void gauss_omp_simd(double** A, double* b, int n, int threads) {
    for (int k = 0; k < n; ++k) {
        double pivot = A[k][k];

        #pragma omp parallel for num_threads(threads)
        for (int i = k + 1; i < n; ++i) {
            double factor = A[i][k] / pivot;
            
            int j = k + 1;
            __m256d v_factor = _mm256_set1_pd(factor);
            for (; j + 3 < n; j += 4) {
                __m256d v_ak = _mm256_loadu_pd(&A[k][j]);
                __m256d v_ai = _mm256_loadu_pd(&A[i][j]); 
                __m256d v_res = _mm256_sub_pd(v_ai, _mm256_mul_pd(v_factor, v_ak));
                _mm256_storeu_pd(&A[i][j], v_res);
            }
            for (; j < n; ++j) {
                A[i][j] -= factor * A[k][j];
            }
            b[i] -= factor * b[k];
        }
    }
}

int main(int argc, char* argv[]) {
    int n = 1024;
    int threads = 4;
    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) threads = atoi(argv[2]);

    srand(2026);

    double** A = alloc_matrix(n);
    double** A_backup = alloc_matrix(n);
    double* b = (double*)aligned_alloc(32, n * sizeof(double));
    double* b_backup = (double*)aligned_alloc(32, n * sizeof(double));
    double* x_true = (double*)aligned_alloc(32, n * sizeof(double));
    double* x = (double*)calloc(n, sizeof(double));

    gen_matrix(A, b, x_true, n);
    
    // 备份以便多次测试
    memcpy(A_backup[0], A[0], n * n * sizeof(double));
    memcpy(b_backup, b, n * sizeof(double));

    printf("=== 规模: %d | 线程数: %d ===\n", n, threads);

    // 1. 测试 Pthread + SIMD
    double start = omp_get_wtime();
    gauss_pthread_simd(A, b, n, threads);
    back_substitution(A, b, n, x); // 🌟 补上回代得到 x
    double t_pthread = omp_get_wtime() - start;
    printf("Pthread + SIMD 耗时: %7.4f 秒 | 校验: %s\n", t_pthread, check_solution(x, x_true, n) ? "OK" : "ERR");

    // 恢复数据
    memcpy(A[0], A_backup[0], n * n * sizeof(double));
    memcpy(b, b_backup, n * sizeof(double));
    memset(x, 0, n * sizeof(double));

    // 2. 测试 OpenMP + SIMD
    start = omp_get_wtime();
    gauss_omp_simd(A, b, n, threads);
    back_substitution(A, b, n, x); // 🌟 补上回代得到 x
    double t_omp = omp_get_wtime() - start;
    printf("OpenMP  + SIMD 耗时: %7.4f 秒 | 校验: %s\n", t_omp, check_solution(x, x_true, n) ? "OK" : "ERR");

    free_matrix(A); free_matrix(A_backup);
    free(b); free(b_backup); free(x_true); free(x);
    return 0;
}