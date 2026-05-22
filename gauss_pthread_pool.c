#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <pthread.h>

// 线程池全局共享控制块
typedef struct {
    int k;              // 当前消去轮次
    int n;              // 矩阵规模
    double** A;
    double* b;
    int t_count;
    int stop;           // 退出标志
    pthread_barrier_t barrier_start;  // 启动消去屏障
    pthread_barrier_t barrier_end;    // 结束消去屏障
} pool_ctrl_t;

static pool_ctrl_t ctrl;

void* thread_pool_worker(void* arg) {
    int t_id = *(int*)arg;
    while (1) {
        // 子线程在轻量屏障处阻塞，等待主线程完成当前轮次主元行的归一化
        pthread_barrier_wait(&ctrl.barrier_start);
        if (ctrl.stop) break;

        int k = ctrl.k;
        int n = ctrl.n;
        double** A = ctrl.A;
        double* b = ctrl.b;

        // 水平块循环划分 (Row-Cyclic) + AVX2+FMA 向量化更新
        for (int i = k + 1 + t_id; i < n; i += ctrl.t_count) {
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
        }

        // 等待所有工作线程完成本轮消去，通知主线程推进
        pthread_barrier_wait(&ctrl.barrier_end);
    }
    return NULL;
}

// 供主程序调用的接口
void gauss_pthread_pool(double** A, double* b, int n, int threads) {
    int* t_ids = (int*)malloc(threads * sizeof(int));
    pthread_t* handles = (pthread_t*)malloc(threads * sizeof(pthread_t));
    
    ctrl.n = n; ctrl.A = A; ctrl.b = b; ctrl.t_count = threads; ctrl.stop = 0;
    pthread_barrier_init(&ctrl.barrier_start, NULL, threads + 1);
    pthread_barrier_init(&ctrl.barrier_end, NULL, threads + 1);

    for (int t = 0; t < threads; ++t) {
        t_ids[t] = t;
        pthread_create(&handles[t], NULL, thread_pool_worker, &t_ids[t]);
    }

    for (int k = 0; k < n; ++k) {
        ctrl.k = k;
        // 唤醒常驻工作线程池
        pthread_barrier_wait(&ctrl.barrier_start);
        // 等待它们并行消去完毕
        pthread_barrier_wait(&ctrl.barrier_end);
    }

    // 退出并销毁线程池
    ctrl.stop = 1;
    pthread_barrier_wait(&ctrl.barrier_start);
    for (int t = 0; t < threads; ++t) pthread_join(handles[t], NULL);
    
    pthread_barrier_destroy(&ctrl.barrier_start);
    pthread_barrier_destroy(&ctrl.barrier_end);
    free(t_ids); free(handles);
}
