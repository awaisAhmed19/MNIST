#include <cuda_runtime.h>
#include <memory>
#include "tensor.h"

//  Synchronization

void TtoDevice(Tensor& t) {
#ifdef USE_CUDA
    if (!t.dirty_device) return;

    CUDA_CHECK(cudaMemcpy(
        t.d_data,
        t.h_data,
        t.rows * t.cols * sizeof(float),
        cudaMemcpyHostToDevice));

    t.dirty_device = false;
#endif
}

void TtoHost(Tensor& t) {
#ifdef USE_CUDA
    if (!t.dirty_host) return;

    CUDA_CHECK(cudaMemcpy(
        t.h_data,
        t.d_data,
        t.rows * t.cols * sizeof(float),
        cudaMemcpyDeviceToHost));

    t.dirty_host = false;
#endif
}

//  Kernels

__global__ void k_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

__global__ void k_sub(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] - b[i];
}

__global__ void k_mul(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * b[i];
}

__global__ void k_matmul(const float* A, const float* B, float* C,
                         int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int p = 0; p < K; ++p)
            sum += A[row * K + p] * B[p * N + col];
        C[row * N + col] = sum;
    }
}

__global__ void k_scale(const float* a, float s, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * s;
}

__global__ void k_sigmoid(const float* x, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = 1.0f / (1.0f + expf(-x[i]));
}

__global__ void k_sigmoid_prime(const float* x, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float s = 1.0f / (1.0f + expf(-x[i]));
        out[i] = s * (1.0f - s);
    }
}

__global__ void k_relu(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = x[i] < 0 ? 0 : x[i];
}

__global__ void k_relu_prime(const float* x, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] > 0 ? 1.0f : 0.0f;
}

__global__ void k_softmax_rows(const float* x, float* out,
                               int rows, int cols) {
    int r = blockIdx.x;
    if (r >= rows) return;

    int base = r * cols;
    float maxv = -INFINITY;

    for (int j = 0; j < cols; j++)
        maxv = fmaxf(maxv, x[base + j]);

    float sum = 0;
    for (int j = 0; j < cols; j++) {
        float e = expf(x[base + j] - maxv);
        out[base + j] = e;
        sum += e;
    }

    for (int j = 0; j < cols; j++)
        out[base + j] /= sum;
}

__global__ void k_softmax_cols(const float* x, float* out,
                               int rows, int cols) {
    int c = blockIdx.x;
    if (c >= cols) return;

    float maxv = -INFINITY;
    for (int r = 0; r < rows; r++)
        maxv = fmaxf(maxv, x[r * cols + c]);

    float sum = 0;
    for (int r = 0; r < rows; r++) {
        float e = expf(x[r * cols + c] - maxv);
        out[r * cols + c] = e;
        sum += e;
    }

    for (int r = 0; r < rows; r++)
        out[r * cols + c] /= sum;
}

__global__ void k_transpose(const float* A, float* B,
                            int rows, int cols) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < rows && c < cols)
        B[c * rows + r] = A[r * cols + c];
}

__global__ void k_update(float* w, const float* grad, float lr, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        w[i] -= lr * grad[i];
}

//  GPU Ops (unique_ptr everywhere)
void TupdateGPU(Tensor& W, const Tensor& dW, float lr)
{
    TtoDevice(W);
    TtoDevice(const_cast<Tensor&>(dW));

    int n = W.size();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    k_update<<<blocks, threads>>>(W.d_data, dW.d_data, lr, n);
    cudaDeviceSynchronize();

    W.dirty_host = true;
}

std::unique_ptr<Tensor> TaddGPU(Tensor& A, Tensor& B) {
    auto C = std::make_unique<Tensor>(A.rows, A.cols);

    TtoDevice(A);
    TtoDevice(B);

    int n = A.rows * A.cols;
    int block = 256;
    int grid  = (n + block - 1) / block;

    k_add<<<grid, block>>>(A.d_data, B.d_data, C->d_data, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    C->dirty_host = true;
    return C;
}

std::unique_ptr<Tensor> TsubGPU(Tensor& A, Tensor& B) {
    auto C = std::make_unique<Tensor>(A.rows, A.cols);

    TtoDevice(A);
    TtoDevice(B);

    int n = A.rows * A.cols;
    int block = 256;
    int grid  = (n + block - 1) / block;

    k_sub<<<grid, block>>>(A.d_data, B.d_data, C->d_data, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    C->dirty_host = true;
    return C;
}

std::unique_ptr<Tensor> TmulGPU(Tensor& A, Tensor& B) {
    auto C = std::make_unique<Tensor>(A.rows, A.cols);

    TtoDevice(A);
    TtoDevice(B);

    int n = A.rows * A.cols;
    int block = 256;
    int grid  = (n + block - 1) / block;

    k_mul<<<grid, block>>>(A.d_data, B.d_data, C->d_data, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    C->dirty_host = true;
    return C;
}

std::unique_ptr<Tensor> TmatmulGPU(Tensor& A, Tensor& B) {
    auto C = std::make_unique<Tensor>(A.rows, B.cols);

    TtoDevice(A);
    TtoDevice(B);

    dim3 block(16,16);
    dim3 grid((B.cols + 15)/16, (A.rows + 15)/16);

    k_matmul<<<grid, block>>>(
        A.d_data, B.d_data, C->d_data,
        A.rows, A.cols, B.cols);

    CUDA_CHECK(cudaDeviceSynchronize());

    C->dirty_host = true;
    return C;
}

std::unique_ptr<Tensor> TscaleGPU(Tensor& A, float s) {
    auto C = std::make_unique<Tensor>(A.rows, A.cols);

    TtoDevice(A);

    int n = A.rows * A.cols;
    int block = 256;
    int grid  = (n + block - 1) / block;

    k_scale<<<grid, block>>>(A.d_data, s, C->d_data, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    C->dirty_host = true;
    return C;
}

std::unique_ptr<Tensor> TSigmoidGPU(Tensor& A) {
    auto C = std::make_unique<Tensor>(A.rows, A.cols);

    TtoDevice(A);

    int n = A.rows * A.cols;
    int block = 256;
    int grid  = (n + block - 1) / block;

    k_sigmoid<<<grid, block>>>(A.d_data, C->d_data, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    C->dirty_host = true;
    return C;
}

std::unique_ptr<Tensor> TSigmoidPrimeGPU(Tensor& A) {
    auto C = std::make_unique<Tensor>(A.rows, A.cols);

    TtoDevice(A);

    int n = A.rows * A.cols;
    int block = 256;
    int grid  = (n + block - 1) / block;

    k_sigmoid_prime<<<grid, block>>>(A.d_data, C->d_data, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    C->dirty_host = true;
    return C;
}

void TReluGPU(Tensor& A) {
    TtoDevice(A);

    int n = A.rows * A.cols;
    int block = 256;
    int grid  = (n + block - 1) / block;

    k_relu<<<grid, block>>>(A.d_data, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    A.dirty_host = true;
}

std::unique_ptr<Tensor> TReluPrimeGPU(Tensor& A) {
    auto C = std::make_unique<Tensor>(A.rows, A.cols);

    TtoDevice(A);

    int n = A.rows * A.cols;
    int block = 256;
    int grid  = (n + block - 1) / block;

    k_relu_prime<<<grid, block>>>(A.d_data, C->d_data, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    C->dirty_host = true;
    return C;
}

std::unique_ptr<Tensor> TSoftmaxRowsGPU(Tensor& A) {
    auto C = std::make_unique<Tensor>(A.rows, A.cols);

    TtoDevice(A);

    k_softmax_rows<<<A.rows, 1>>>(
        A.d_data, C->d_data, A.rows, A.cols);

    CUDA_CHECK(cudaDeviceSynchronize());

    C->dirty_host = true;
    return C;
}

std::unique_ptr<Tensor> TSoftmaxColsGPU(Tensor& A) {
    auto C = std::make_unique<Tensor>(A.rows, A.cols);

    TtoDevice(A);

    k_softmax_cols<<<A.cols, 1>>>(
        A.d_data, C->d_data, A.rows, A.cols);

    CUDA_CHECK(cudaDeviceSynchronize());

    C->dirty_host = true;
    return C;
}

std::unique_ptr<Tensor> TtransposeGPU(Tensor& A) {
    auto C = std::make_unique<Tensor>(A.cols, A.rows);

    TtoDevice(A);

    dim3 block(16,16);
    dim3 grid((A.cols+15)/16, (A.rows+15)/16);

    k_transpose<<<grid, block>>>(
        A.d_data, C->d_data, A.rows, A.cols);

    CUDA_CHECK(cudaDeviceSynchronize());

    C->dirty_host = true;
    return C;
}
