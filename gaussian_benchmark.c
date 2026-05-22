#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h> // 用于 AVX2 SIMD 加速
#include <pthread.h>   // Pthread 支持
#include <omp.h>       // OpenMP 支持

// 全局变量定义
int N = 2048; 
float *mat_serial = NULL;
float *mat_pthread = NULL;
float *mat_omp = NULL;
float *mat_pool = NULL;
float *mat_pipeline = NULL;

// Pthread 传参结构体
typedef struct {
    int thread_id;
    int total_threads;
} thread_data_t;

pthread_barrier_t barrier; // 基础 Pthread 屏障

// ==========================================
// 进阶一：常驻工作线程池控制块与函数
// ==========================================
typedef struct {
    int k;
    int stop;
    int t_count;
    pthread_barrier_t barrier_start;
    pthread_barrier_t barrier_end;
} pool_ctrl_t;

static pool_ctrl_t pool_ctrl;

void* thread_pool_worker(void* arg) {
    int t_id = *(int*)arg;
    while (1) {
        pthread_barrier_wait(&pool_ctrl.barrier_start);
        if (pool_ctrl.stop) break;

        int k = pool_ctrl.k;
        for (int i = k + 1 + t_id; i < N; i += pool_ctrl.t_count) {
            float tmp = mat_pool[i * N + k];
            int j_simd = k + 1;
            __m256 v_tmp = _mm256_set1_ps(tmp);
            for (; j_simd <= N - 8; j_simd += 8) {
                __m256 v_mat_k = _mm256_loadu_ps(&mat_pool[k * N + j_simd]);
                __m256 v_mat_i = _mm256_loadu_ps(&mat_pool[i * N + j_simd]);
                __m256 v_res = _mm256_sub_ps(v_mat_i, _mm256_mul_ps(v_tmp, v_mat_k));
                _mm256_storeu_ps(&mat_pool[i * N + j_simd], v_res);
            }
            for (; j_simd < N; ++j_simd) {
                mat_pool[i * N + j_simd] -= tmp * mat_pool[k * N + j_simd];
            }
            mat_pool[i * N + k] = 0.0f;
        }
        pthread_barrier_wait(&pool_ctrl.barrier_end);
    }
    return NULL;
}

void run_pthread_pool_gaussian(int threads) {
    pthread_t* handles = malloc(sizeof(pthread_t) * threads);
    int* t_ids = malloc(sizeof(int) * threads);
    
    pool_ctrl.stop = 0; pool_ctrl.t_count = threads;
    pthread_barrier_init(&pool_ctrl.barrier_start, NULL, threads + 1);
    pthread_barrier_init(&pool_ctrl.barrier_end, NULL, threads + 1);

    for (int t = 0; t < threads; t++) {
        t_ids[t] = t;
        pthread_create(&handles[t], NULL, thread_pool_worker, &t_ids[t]);
    }

    for (int k = 0; k < N; ++k) {
        // 主线程负责归一化
        float tmp = mat_pool[k * N + k];
        int j_simd = k + 1;
        __m256 v_tmp = _mm256_set1_ps(tmp);
        for (; j_simd <= N - 8; j_simd += 8) {
            __m256 v_mat = _mm256_loadu_ps(&mat_pool[k * N + j_simd]);
            _mm256_storeu_ps(&mat_pool[k * N + j_simd], _mm256_div_ps(v_mat, v_tmp));
        }
        for (; j_simd < N; ++j_simd) mat_pool[k * N + j_simd] /= tmp;
        mat_pool[k * N + k] = 1.0f;

        pool_ctrl.k = k;
        pthread_barrier_wait(&pool_ctrl.barrier_start); // 唤醒线程池
        pthread_barrier_wait(&pool_ctrl.barrier_end);   // 等待消去完
    }

    pool_ctrl.stop = 1;
    pthread_barrier_wait(&pool_ctrl.barrier_start);
    for (int t = 0; t < threads; t++) pthread_join(handles[t], NULL);
    
    pthread_barrier_destroy(&pool_ctrl.barrier_start);
    pthread_barrier_destroy(&pool_ctrl.barrier_end);
    free(handles); free(t_ids);
}

// ==========================================
// 进阶二：前瞻非阻塞流水线控制块与函数
// ==========================================
static volatile int* row_ready = NULL;

typedef struct {
    int t_id;
    int t_count;
} pipe_data_t;

void* pipeline_worker(void* arg) {
    pipe_data_t* data = (pipe_data_t*)arg;
    int t_id = data->t_id;
    int t_count = data->t_count;

    for (int i = t_id; i < N; i += t_count) {
        for (int k = 0; k < i; ++k) {
            // 异步流水线等待主元行准备就绪
            while (row_ready[k] < k) {
                #if defined(__x86_64__)
                __builtin_ia32_pause();
                #endif
            }

            float tmp = mat_pipeline[i * N + k];
            int j_simd = k + 1;
            __m256 v_tmp = _mm256_set1_ps(tmp);
            for (; j_simd <= N - 8; j_simd += 8) {
                __m256 v_mat_k = _mm256_loadu_ps(&mat_pipeline[k * N + j_simd]);
                __m256d v_mat_k_l = _mm256_cvtps_pd(_mm256_extractf128_ps(v_mat_k, 0)); // 转换兼容
                __m256 v_mat_i = _mm256_loadu_ps(&mat_pipeline[i * N + j_simd]);
                __m256 v_res = _mm256_sub_ps(v_mat_i, _mm256_mul_ps(v_tmp, v_mat_k));
                _mm256_storeu_ps(&mat_pipeline[i * N + j_simd], v_res);
            }
            for (; j_simd < N; ++j_simd) {
                mat_pipeline[i * N + j_simd] -= tmp * mat_pipeline[k * N + j_simd];
            }
            mat_pipeline[i * N + k] = 0.0f;

            if (i == k + 1) {
                // 就地完成归一化
                float p_tmp = mat_pipeline[i * N + i];
                int jj = i + 1;
                __m256 v_ptmp = _mm256_set1_ps(p_tmp);
                for (; jj <= N - 8; jj += 8) {
                    __m256 v_m = _mm256_loadu_ps(&mat_pipeline[i * N + jj]);
                    _mm256_storeu_ps(&mat_pipeline[i * N + jj], _mm256_div_ps(v_m, v_ptmp));
                }
                for (; jj < N; ++jj) mat_pipeline[i * N + jj] /= p_tmp;
                mat_pipeline[i * N + i] = 1.0f;
                row_ready[i] = i; // 释放状态
            }
        }
    }
    return NULL;
}

void run_pipeline_gaussian(int threads) {
    row_ready = malloc(sizeof(int) * N);
    for (int i = 0; i < N; i++) row_ready[i] = -1;

    // 处理第 0 行初始归一化
    float tmp = mat_pipeline[0];
    int j_simd = 1; __m256 v_tmp = _mm256_set1_ps(tmp);
    for (; j_simd <= N - 8; j_simd += 8) {
        __m256 v_mat = _mm256_loadu_ps(&mat_pipeline[j_simd]);
        _mm256_storeu_ps(&mat_pipeline[j_simd], _mm256_div_ps(v_mat, v_tmp));
    }
    for (; j_simd < N; ++j_simd) mat_pipeline[j_simd] /= tmp;
    mat_pipeline[0] = 1.0f;
    row_ready[0] = 0;

    pthread_t* handles = malloc(sizeof(pthread_t) * threads);
    pipe_data_t* data = malloc(sizeof(pipe_data_t) * threads);
    for (int t = 0; t < threads; t++) {
        data[t].thread_id = t; data[t].total_threads = threads;
        pthread_create(&handles[t], NULL, pipeline_worker, &data[t]);
    }
    for (int t = 0; t < threads; t++) pthread_join(handles[t], NULL);
    free(handles); free(data); free((void*)row_ready);
}

// 矩阵动态内存分配与初始化
void init_matrices(int size) {
    N = size;
    mat_serial = (float *)malloc(sizeof(float) * N * N);
    mat_pthread = (float *)malloc(sizeof(float) * N * N);
    mat_omp = (float *)malloc(sizeof(float) * N * N);
    mat_pool = (float *)malloc(sizeof(float) * N * N);
    mat_pipeline = (float *)malloc(sizeof(float) * N * N);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float val = (float)rand() / RAND_MAX;
            mat_serial[i * N + j] = val;
            mat_pthread[i * N + j] = val;
            mat_omp[i * N + j] = val;
            mat_pool[i * N + j] = val;
            mat_pipeline[i * N + j] = val;
        }
        mat_serial[i * N + i] += N;
        mat_pthread[i * N + i] += N;
        mat_omp[i * N + i] += N;
        mat_pool[i * N + i] += N;
        mat_pipeline[i * N + i] += N;
    }
}

void free_matrices() {
    free(mat_serial); free(mat_pthread); free(mat_omp);
    free(mat_pool); free(mat_pipeline);
}

// 1. 纯串行高斯消去
void serial_gaussian() {
    for (int k = 0; k < N; ++k) {
        float tmp = mat_serial[k * N + k];
        for (int j = k + 1; j < N; ++j) mat_serial[k * N + j] /= tmp;
        mat_serial[k * N + k] = 1.0f;
        for (int i = k + 1; i < N; ++i) {
            tmp = mat_serial[i * N + k];
            for (int j = k + 1; j < N; ++j) mat_serial[i * N + j] -= tmp * mat_serial[k * N + j];
            mat_serial[i * N + k] = 0.0f;
        }
    }
}

// 2. 基础 Pthread 算法
void* pthread_worker(void* arg) {
    thread_data_t* data = (thread_data_t*)arg;
    int t_id = data->thread_id; int t_count = data->total_threads;
    for (int k = 0; k < N; ++k) {
        if (t_id == 0) {
            float tmp = mat_pthread[k * N + k];
            int j_simd = k + 1; __m256 v_tmp = _mm256_set1_ps(tmp);
            for (; j_simd <= N - 8; j_simd += 8) {
                __m256 v_mat = _mm256_loadu_ps(&mat_pthread[k * N + j_simd]);
                _mm256_storeu_ps(&mat_pthread[k * N + j_simd], _mm256_div_ps(v_mat, v_tmp));
            }
            for (; j_simd < N; ++j_simd) mat_pthread[k * N + j_simd] /= tmp;
            mat_pthread[k * N + k] = 1.0f;
        }
        pthread_barrier_wait(&barrier);
        for (int i = k + 1 + t_id; i < N; i += t_count) {
            float tmp = mat_pthread[i * N + k];
            int j_simd = k + 1; __m256 v_tmp = _mm256_set1_ps(tmp);
            for (; j_simd <= N - 8; j_simd += 8) {
                __m256 v_mat_k = _mm256_loadu_ps(&mat_pthread[k * N + j_simd]);
                __m256 v_mat_i = _mm256_loadu_ps(&mat_pthread[i * N + j_simd]);
                __m256 v_res = _mm256_sub_ps(v_mat_i, _mm256_mul_ps(v_tmp, v_mat_k));
                _mm256_storeu_ps(&mat_pthread[i * N + j_simd], v_res);
            }
            for (; j_simd < N; ++j_simd) mat_pthread[i * N + j_simd] -= tmp * mat_pthread[k * N + j_simd];
            mat_pthread[i * N + k] = 0.0f;
        }
        pthread_barrier_wait(&barrier);
    }
    pthread_exit(NULL);
}

void run_pthread_gaussian(int threads) {
    pthread_t* handles = malloc(sizeof(pthread_t) * threads);
    thread_data_t* data = malloc(sizeof(thread_data_t) * threads);
    pthread_barrier_init(&barrier, NULL, threads);
    for (int t = 0; t < threads; t++) {
        data[t].thread_id = t; data[t].total_threads = threads;
        pthread_create(&handles[t], NULL, pthread_worker, (void*)&data[t]);
    }
    for (int t = 0; t < threads; t++) pthread_join(handles[t], NULL);
    pthread_barrier_destroy(&barrier); free(handles); free(data);
}

// 3. 基础 OpenMP 算法
void run_omp_gaussian(int threads) {
    int i, j, k; float tmp;
    #pragma omp parallel num_threads(threads) private(i, j, k, tmp)
    {
        for (k = 0; k < N; ++k) {
            #pragma omp single
            {
                tmp = mat_omp[k * N + k]; int j_simd = k + 1; __m256 v_tmp = _mm256_set1_ps(tmp);
                for (; j_simd <= N - 8; j_simd += 8) {
                    __m256 v_mat = _mm256_loadu_ps(&mat_omp[k * N + j_simd]);
                    _mm256_storeu_ps(&mat_omp[k * N + j_simd], _mm256_div_ps(v_mat, v_tmp));
                }
                for (; j_simd < N; ++j_simd) mat_omp[k * N + j_simd] /= tmp;
                mat_omp[k * N + k] = 1.0f;
            }
            #pragma omp for schedule(static, 1)
            for (i = k + 1; i < N; ++i) {
                tmp = mat_omp[i * N + k]; int j_simd = k + 1; __m256 v_tmp = _mm256_set1_ps(tmp);
                for (; j_simd <= N - 8; j_simd += 8) {
                    __m256 v_mat_k = _mm256_loadu_ps(&mat_omp[k * N + j_simd]);
                    __m256 v_mat_i = _mm256_loadu_ps(&mat_omp[i * N + j_simd]);
                    __m256 v_res = _mm256_sub_ps(v_mat_i, _mm256_mul_ps(v_tmp, v_mat_k));
                    _mm256_storeu_ps(&mat_omp[i * N + j_simd], v_res);
                }
                for (; j_simd < N; ++j_simd) mat_omp[i * N + j_simd] -= tmp * mat_omp[k * N + j_simd];
                mat_omp[i * N + k] = 0.0f;
            }
        }
    }
}

// 结果验证
int verify_results() {
    for (int i = 0; i < N * N; i++) {
        if (fabs(mat_serial[i] - mat_pthread[i]) > 1e-2) return 0;
        if (fabs(mat_serial[i] - mat_omp[i]) > 1e-2) return 0;
        if (fabs(mat_serial[i] - mat_pool[i]) > 1e-2) return 0;
        if (fabs(mat_serial[i] - mat_pipeline[i]) > 1e-2) return 0;
    }
    return 1;
}

int main(int argc, char* argv[]) {
    int size = 1024; int threads = 4;
    if (argc > 1) size = atoi(argv[1]);
    if (argc > 2) threads = atoi(argv[2]);

    init_matrices(size);
    printf("规模: %4d | 线程数: %2d \n", size, threads);

    double start = omp_get_wtime();
    serial_gaussian();
    double t_serial = omp_get_wtime() - start;
    printf("-> [串行方案]: %7.4f 秒\n", t_serial);

    start = omp_get_wtime();
    run_pthread_gaussian(threads);
    double t_pthread = omp_get_wtime() - start;
    printf("-> [基础Pthread]: %7.4f 秒 (加速比: %.2fx)\n", t_pthread, t_serial / t_pthread);

    start = omp_get_wtime();
    run_omp_gaussian(threads);
    double t_omp = omp_get_wtime() - start;
    printf("-> [基础OpenMP]:  %7.4f 秒 (加速比: %.2fx)\n", t_omp, t_serial / t_omp);

    start = omp_get_wtime();
    run_pthread_pool_gaussian(threads);
    double t_pool = omp_get_wtime() - start;
    printf("-> [进阶线程池]: %7.4f 秒 (加速比: %.2fx)\n", t_pool, t_serial / t_pool);

    start = omp_get_wtime();
    run_pipeline_gaussian(threads);
    double t_pipe = omp_get_wtime() - start;
    printf("-> [进阶流水线]: %7.4f 秒 (加速比: %.2fx)\n", t_pipe, t_serial / t_pipe);

    if (verify_results()) printf("| 验证结果: [OK] 所有矩阵计算完全一致\n");
    else printf("| 验证结果: [ERR] 发生数值偏差\n");

    free_matrices();
    return 0;
}