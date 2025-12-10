#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include "tensor.h"

#define CUDA_CHECK(err) \
    do { if ((err) != cudaSuccess) { \
        fprintf(stderr, "CUDA ERROR %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); } } while(0)

inline int Tsize(const Tensor* t) { return t->rows * t->cols; }
//---memory-----
Tensor* Tcreate(int r, int c) {
    if (r <= 0 || c <= 0) return nullptr;
    Tensor* t = (Tensor*) malloc(sizeof(Tensor));
    t->rows = r;
    t->cols = c;
    t->dirty_host = false; t->dirty_device = false;
    int n = r * c;
    t->h_data = (float*) malloc(n * sizeof(float));
    if (!t->h_data) { free(t); return nullptr; }
    std::fill(t->h_data, t->h_data + n, 0.0f);
    CUDA_CHECK(cudaMalloc(&t->d_data, n * sizeof(float)));
    CUDA_CHECK(cudaMemset(t->d_data, 0, n * sizeof(float)));
    return t;
}
void Tfree(Tensor* t) {
    if (!t) return;
    free(t->h_data);
    CUDA_CHECK(cudaFree(t->d_data));
    free(t);
}

void TtoDevice(Tensor* t) {
    int n = Tsize(t);
    CUDA_CHECK(cudaMemcpy(t->d_data, t->h_data, n * sizeof(float), cudaMemcpyHostToDevice));
    t->dirty_device = false;
}
void TtoHost(Tensor* t) {
    int n = Tsize(t);
    CUDA_CHECK(cudaMemcpy(t->h_data, t->d_data, n * sizeof(float), cudaMemcpyDeviceToHost));
    t->dirty_host = false;
}


//-----------kernels---------

__global__ void k_add(const float* a, const float* b, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] + b[idx];
}

Tensor* TaddGPU(Tensor* A, Tensor* B) {
    if (!A || !B) return nullptr;
    if (A->rows != B->rows || A->cols != B->cols) return nullptr;
    int n = Tsize(A);
    Tensor* C = Tcreate(A->rows, A->cols);

    if (A->dirty_device) TtoDevice(A);
    if (B->dirty_device) TtoDevice(B);

    int block = 256;
    int grid = (n + block - 1) / block;
    k_add<<<grid, block>>>(A->d_data, B->d_data, C->d_data, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    C->dirty_device = false; // C contains current device data
    C->dirty_host = true;    // host copy is stale
    return C;
}
__global__ void k_sub(const float* a, const float* b, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] - b[idx];
}

Tensor* TsubGPU(Tensor* A, Tensor* B) {
    if (!A || !B) return nullptr;
    if (A->rows != B->rows || A->cols != B->cols) return nullptr;

    int n = Tsize(A);
    Tensor* C = Tcreate(A->rows, A->cols);

    if (A->dirty_device) TtoDevice(A);
    if (B->dirty_device) TtoDevice(B);

    int block = 256;
    int grid = (n + block - 1) / block;

    k_sub<<<grid, block>>>(A->d_data, B->d_data, C->d_data, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    C->dirty_device = false;  // device data is clean
    C->dirty_host = true;     // host data outdated

    return C;
}

__global__ void k_matmul(const float* A, const float* B, float* C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int p = 0; p < K; ++p) {
            sum += A[row * K + p] * B[p * N + col];
        }
        C[row * N + col] = sum;
    }
}

Tensor* TmatmulGPU(Tensor* A, Tensor* B) {
    if (!A || !B) return nullptr;
    int M = A->rows, K = A->cols, K2 = B->rows, N = B->cols;
    if (K != K2) return nullptr;
    Tensor* C = Tcreate(M, N);

    if (A->dirty_device) TtoDevice(A);
    if (B->dirty_device) TtoDevice(B);

    dim3 block(16,16);
    dim3 grid((N+15)/16, (M+15)/16);
    k_matmul<<<grid, block>>>(A->d_data, B->d_data, C->d_data, M, K, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    C->dirty_device = false;
    C->dirty_host = true;
    return C;
}
