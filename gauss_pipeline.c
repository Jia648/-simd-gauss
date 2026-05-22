#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <pthread.h>

// row_ready[i] = k 表示第 i 行当前已经接收并完成了前 k 轮主元的消去更新
// 当 row_ready[k] == k 时，说明第 k 行本身已经就绪，可以作为主元行去消去更下方的行
static volatile int* row_ready;

typedef struct {
    int t_id; int t_count; int n; double** A; double* b;
} pipe_args_t;

void* pipeline_worker_thread(void* arg) {
    pipe_args_t* args = (pipe_args_t*)arg;
    int n = args->n; double** A = args->A; double* b = args->b;
    int t_id = args->t_id; int t_count = args->t_count;

    // 水平交替认领固定的物理行
    for (int i = t_id; i < n; i += t_count) {
        for (int k = 0; k < i; ++k) {
            // 异步流水线等待：只要主元行 k 没准备好，就在用户态自旋，绝不让出内核态
            while (row_ready[k] < k) {
                #if defined(__x86_64__)
                __builtin_ia32_pause(); // 低功耗自旋提示
                #endif
            }

            // SIMD 纵向混合消去
            double factor = A[i][k] / A[k][k];
            int j = k + 1;
            __m256d v_factor = _mm256_set1_pd(factor);
            for (; j + 3 < n; j += 4) {
                __m256d v_ak = _mm256_loadu_pd(&A[k][j]);
                __m256d v_ai = _mm256_loadu_pd(&A[i][j]);
                __m256d v_res = _mm256_sub_pd(v_ai, _mm256_mul_pd(v_factor, v_ak));
                _mm256_storeu_pd(&A[i][j], v_res);
            }
            for (; j < n; ++j) A[i][j] -= factor * A[k][j];
            b[i] -= factor * b[k];

            // 核心前瞻：如果当前行消去完成，且它刚好是下一轮的主元行 (i == k + 1)
            if (i == k + 1) {
                row_ready[i] = i; // 瞬间解锁下方正在自旋等待该主元的其他线程
            }
        }
    }
    return NULL;
}

// 供主程序调用的接口
void gauss_pipeline(double** A, double* b, int n, int threads) {
    row_ready = (volatile int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; ++i) row_ready[i] = -1;

    pthread_t* handles = (pthread_t*)malloc(threads * sizeof(pthread_t));
    pipe_args_t* args = (pipe_args_t*)malloc(threads * sizeof(pipe_args_t));

    for (int t = 0; t < threads; ++t) {
        args[t].t_id = t;
        args[t].t_count = threads;
        args[t].n = n;
        args[t].A = A;
        args[t].b = b;
        pthread_create(&handles[t], NULL, pipeline_worker_thread, &args[t]);
    }

    for (int t = 0; t < threads; ++t) {
        pthread_join(handles[t], NULL);
    }

    free((void*)row_ready);
    free(handles);
    free(args);
}
