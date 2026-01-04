#include <cuda_runtime.h>
#include <cstdlib>
#include <algorithm>
#include "tensor.h"

//---memory-----
Tensor* TcreateGPU(int r, int c) {
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

Tensor* TcopyGPU(const Tensor* src) {
    if (!src) return nullptr;

    Tensor* out = TcreateGPU(src->rows, src->cols);

    if (src->dirty_device)
        TtoDevice(const_cast<Tensor*>(src));

    int n = Tsize(src);

    CUDA_CHECK(cudaMemcpy(
        out->d_data,
        src->d_data,
        n * sizeof(float),
        cudaMemcpyDeviceToDevice
    ));

    out->dirty_device = false;
    out->dirty_host   = true;

    return out;
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

Tensor* TaddGPU(const Tensor* A, const Tensor* B) {
    if (!A || !B) return nullptr;
    if (A->rows != B->rows || A->cols != B->cols) return nullptr;
    int n = Tsize(A);
    Tensor* C = TcreateGPU(A->rows, A->cols);



if (A->dirty_device)
    TtoDevice(const_cast<Tensor*>(A));

if (B->dirty_device)
    TtoDevice(const_cast<Tensor*>(B));


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

Tensor* TsubGPU(const Tensor* A, const Tensor* B) {
    if (!A || !B) return nullptr;
    if (A->rows != B->rows || A->cols != B->cols) return nullptr;

    int n = Tsize(A);
    Tensor* C = TcreateGPU(A->rows, A->cols);


if (A->dirty_device)
    TtoDevice(const_cast<Tensor*>(A));

if (B->dirty_device)
    TtoDevice(const_cast<Tensor*>(B));


    int block = 256;
    int grid = (n + block - 1) / block;

    k_sub<<<grid, block>>>(A->d_data, B->d_data, C->d_data, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    C->dirty_device = false;  // device data is clean
    C->dirty_host = true;     // host data outdated

    return C;
}

__global__ void k_mul(const float* a, const float* b, float* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = a[i] * b[i];
}

Tensor* TmulGPU(const Tensor* A,const Tensor* B)
{
    if (!A || !B) return nullptr;
    if (A->rows != B->rows || A->cols != B->cols) return nullptr;

    int n = Tsize(A);
    Tensor* C = TcreateGPU(A->rows, A->cols);

if (A->dirty_device)
    TtoDevice(const_cast<Tensor*>(A));

if (B->dirty_device)
    TtoDevice(const_cast<Tensor*>(B));


    int block = 256;
    int grid  = (n + block - 1) / block;

    k_mul<<<grid, block>>>(A->d_data, B->d_data, C->d_data, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    C->dirty_device = false;
    C->dirty_host   = true;

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

Tensor* TmatmulGPU(const Tensor* A, const Tensor* B) {
    if (!A || !B) return nullptr;
    int M = A->rows, K = A->cols, K2 = B->rows, N = B->cols;
    if (K != K2) return nullptr;
    Tensor* C = TcreateGPU(M, N);


if (A->dirty_device)
    TtoDevice(const_cast<Tensor*>(A));

if (B->dirty_device)
    TtoDevice(const_cast<Tensor*>(B));


    dim3 block(16,16);
    dim3 grid((N+15)/16, (M+15)/16);
    k_matmul<<<grid, block>>>(A->d_data, B->d_data, C->d_data, M, K, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    C->dirty_device = false;
    C->dirty_host = true;
    return C;
}

__global__ void k_scale(const float* a, float s, float* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = a[i] * s;
}

Tensor* TscaleGPU(const Tensor* A, float s)
{
    if (!A) return nullptr;

    int n = Tsize(A);
    Tensor* C = TcreateGPU(A->rows, A->cols);

if (A->dirty_device)
    TtoDevice(const_cast<Tensor*>(A));




    int block = 256;
    int grid  = (n + block - 1) / block;

    k_scale<<<grid, block>>>(A->d_data, s, C->d_data, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    C->dirty_device = false;
    C->dirty_host   = true;

    return C;
}
//activation functions

__global__ void k_sigmoid(float* x, float* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = 1.0f / (1.0f + expf(-x[i]));
}

Tensor* TSigmoidGPU(const Tensor* A)
{
    int n = Tsize(A);
    Tensor* C = TcreateGPU(A->rows, A->cols);

if (A->dirty_device)
    TtoDevice(const_cast<Tensor*>(A));


    int block = 256;
    int grid  = (n + block - 1) / block;

    k_sigmoid<<<grid, block>>>(A->d_data, C->d_data, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    C->dirty_device = false;
    C->dirty_host   = true;

    return C;
}
__global__ void k_sigmoid_prime(const float* x, float* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        float s = 1.0f / (1.0f + expf(-x[i]));
        out[i]  = s * (1.0f - s);
    }
}

Tensor* TSigmoidPrimeGPU(const Tensor* A)
{
    int n = Tsize(A);
    Tensor* C = TcreateGPU(A->rows, A->cols);

if (A->dirty_device)
    TtoDevice(const_cast<Tensor*>(A));

    int block = 256;
    int grid  = (n + block - 1) / block;

    k_sigmoid_prime<<<grid, block>>>(A->d_data, C->d_data, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    C->dirty_device = false;
    C->dirty_host   = true;

    return C;
}
__global__ void k_relu(float* x, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        x[i] = (x[i] < 0.0f ? 0.0f : x[i]);
}

void TReluGPU(Tensor* A)
{
   if (A->dirty_device)
    TtoDevice(const_cast<Tensor*>(A));

    int n = Tsize(A);
    int block = 256;
    int grid  = (n + block - 1) / block;

    k_relu<<<grid, block>>>(A->d_data, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    A->dirty_device = false;
    A->dirty_host   = true;
}
__global__ void k_relu_prime(const float* x, float* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = (x[i] > 0.0f ? 1.0f : 0.0f);
}

Tensor* TReluPrimeGPU(Tensor* A)
{
    if (A->dirty_device)
    TtoDevice(const_cast<Tensor*>(A));

    int n = Tsize(A);
    Tensor* C = TcreateGPU(A->rows, A->cols);

    int block = 256;
    int grid  = (n + block - 1) / block;

    k_relu_prime<<<grid, block>>>(A->d_data, C->d_data, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    C->dirty_device = false;
    C->dirty_host   = true;

    return C;
}
__global__ void k_softmax_rows(float* x, float* out, int rows, int cols)
{
    int r = blockIdx.x;
    if (r >= rows) return;

    float maxv = -INFINITY;
    int base = r * cols;

    for (int j = 0; j < cols; j++)
        maxv = fmaxf(maxv, x[base + j]);

    float sum = 0.0f;
    for (int j = 0; j < cols; j++) {
        float e = expf(x[base + j] - maxv);
        out[base + j] = e;
        sum += e;
    }

    for (int j = 0; j < cols; j++)
        out[base + j] /= sum;
}

Tensor* TSoftmaxRowsGPU(const Tensor* A)
{
    Tensor* C = TcreateGPU(A->rows, A->cols);
    if (A->dirty_device)
    TtoDevice(const_cast<Tensor*>(A));

    k_softmax_rows<<<A->rows, 1>>>(A->d_data, C->d_data, A->rows, A->cols);
    CUDA_CHECK(cudaDeviceSynchronize());

    C->dirty_device = false;
    C->dirty_host   = true;

    return C;
}
__global__ void k_softmax_cols(float* x, float* out, int rows, int cols)
{
    int c = blockIdx.x;
    if (c >= cols) return;

    float maxv = -INFINITY;

    for (int r = 0; r < rows; r++)
        maxv = fmaxf(maxv, x[r * cols + c]);

    float sum = 0.0f;
    for (int r = 0; r < rows; r++) {
        float e = expf(x[r * cols + c] - maxv);
        out[r * cols + c] = e;
        sum += e;
    }

    for (int r = 0; r < rows; r++)
        out[r * cols + c] /= sum;
}

Tensor* TSoftmaxColsGPU(const Tensor* A)
{
    Tensor* C = TcreateGPU(A->rows, A->cols);
    if (A->dirty_device)
    TtoDevice(const_cast<Tensor*>(A));
  k_softmax_cols<<<A->cols, 1>>>(A->d_data, C->d_data, A->rows, A->cols);
    CUDA_CHECK(cudaDeviceSynchronize());

    C->dirty_device = false;
    C->dirty_host   = true;

    return C;
}
__global__ void k_transpose(const float* A, float* B, int rows, int cols)
{
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < rows && c < cols)
        B[c * rows + r] = A[r * cols + c];
}

Tensor* TtransposeGPU(const Tensor* A)
{
    Tensor* C = TcreateGPU(A->cols, A->rows);
   if (A->dirty_device)
    TtoDevice(const_cast<Tensor*>(A));

    dim3 block(16, 16);
    dim3 grid((A->cols + 15) / 16, (A->rows + 15) / 16);

    k_transpose<<<grid, block>>>(A->d_data, C->d_data, A->rows, A->cols);
    CUDA_CHECK(cudaDeviceSynchronize());

    C->dirty_device = false;
    C->dirty_host   = true;

    return C;
}
