#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

// 分配二维矩阵（连续内存+行指针）
double** alloc_matrix(int n) {
    double* data = (double*)malloc(n * n * sizeof(double));
    double** A = (double*)malloc(n * sizeof(double*));
    if (!data || !A) exit(1);
    for (int i = 0; i < n; ++i) A[i] = data + i * n;
    return A;
}

void free_matrix(double** A) {
    if (A) {
        free(A[0]);  // 释放连续数据区
        free(A);     // 释放行指针数组
    }
}

// 生成严格对角占优矩阵和真解
void gen_matrix(double** A, double* b, double* x_true, int n) {
    for (int i = 0; i < n; ++i) {
        double row_sum = 0.0;
        for (int j = 0; j < n; ++j) {
            if (i != j) {
                A[i][j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0; // [-1,1]
                row_sum += fabs(A[i][j]);
            }
        }
        A[i][i] = row_sum + 1.0; // 严格对角占优
    }
    // 随机真解
    for (int i = 0; i < n; ++i)
        x_true[i] = ((double)rand() / RAND_MAX) * 10.0;
    // 计算 b = A * x_true
    memset(b, 0, n * sizeof(double));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            b[i] += A[i][j] * x_true[j];
}

// 验证残差 ||b - A*x||
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

// 串行高斯消去（无选主元）
void gauss_serial(double** A, double* b, int n, double* x) {
    // 前向消去
    for (int k = 0; k < n; ++k) {
        double pivot = A[k][k];
        for (int i = k+1; i < n; ++i) {
            double factor = A[i][k] / pivot;
            for (int j = k+1; j < n; ++j)
                A[i][j] -= factor * A[k][j];
            b[i] -= factor * b[k];
            // A[i][k] = 0.0; // 可选，不设置也不影响后续
        }
    }
    // 回代
    for (int i = n-1; i >= 0; --i) {
        double sum = b[i];
        for (int j = i+1; j < n; ++j)
            sum -= A[i][j] * x[j];
        x[i] = sum / A[i][i];
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: %s <matrix_size>\n", argv[0]);
        return 1;
    }
    int n = atoi(argv[1]);
    srand(2024);

    // 分配原始矩阵和备份
    double** A = alloc_matrix(n);
    double** A_bak = alloc_matrix(n);
    double* b = (double*)malloc(n * sizeof(double));
    double* b_bak = (double*)malloc(n * sizeof(double));
    double* x_true = (double*)malloc(n * sizeof(double));
    double* x = (double*)calloc(n, sizeof(double));

    gen_matrix(A, b, x_true, n);

    // 备份
    memcpy(A_bak[0], A[0], n * n * sizeof(double));
    memcpy(b_bak, b, n * sizeof(double));

    double start = get_time();
    gauss_serial(A, b, n, x);
    double end = get_time();
    double elapsed = end - start;

    // 用原始矩阵和 b 验证残差
    double residual = check_residual(A_bak, b_bak, x, n);
    printf("n = %d, time = %.6f s, residual = %.2e\n", n, elapsed, residual);

    free_matrix(A);
    free_matrix(A_bak);
    free(b);
    free(b_bak);
    free(x_true);
    free(x);
    return 0;
}
