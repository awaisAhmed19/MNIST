#include "matrix.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define TILE 16

// ------------------------
// CUDA KERNELS
// ------------------------

template<typename Tp>
__global__ void elementwiseMulKernel(const Tp* A, const Tp* B, Tp* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) C[idx] = A[idx] * B[idx];
}

template<typename Tp>
__global__ void elementwiseAddKernel(const Tp* A, const Tp* B, Tp* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) C[idx] = A[idx] + B[idx];
}

template<typename Tp>
__global__ void elementwiseScaleKernel(Tp* A, Tp num, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) A[idx] = A[idx]*num;
}

template<typename Tp>
__global__ void elementwiseSubKernel(const Tp* A, const Tp* B, Tp* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) C[idx] = A[idx] - B[idx];
}

template<typename Tp>
__global__ void expKernel(const Tp* in, Tp* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) out[idx] = exp(in[idx]);
}

template<typename Tp>
__global__ void divKernel(const Tp* in, Tp* out, Tp total, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) out[idx] = in[idx] / total;
}

template<typename Tp>
__global__ void sumKernel(const Tp* in, Tp* total, int size) {
    __shared__ Tp cache[256];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    cache[tid] = (idx < size ? in[idx] : 0.0f);

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) cache[tid] += cache[tid + s];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(total, cache[0]);
}

template<typename Tp>
__global__ void fillKernel(Tp* data, Tp value, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = value;
}

template<typename Tp>
__global__ void reluKernel(Tp* in, Tp* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        out[idx] = fmaxf(0.0f, in[idx]);
}
template<typename Tp>
__global__ void reluPrimeKernel(Tp* input, Tp* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        output[idx] = (input[idx] > 0.0f) ? 1.0f : 0.0f;
}

template<typename Tp>
__global__ void tanhKernel(Tp* in, Tp* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        out[idx] = tanhf(in[idx]);
}
template<typename Tp>
__global__ void matmulKernel(const Tp* A, const Tp* B, Tp* C, int A_rows, int A_cols, int B_cols)
{
    __shared__ Tp tileA[TILE][TILE];
    __shared__ Tp tileB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    Tp sum = 0;

    int tiles = (A_cols + TILE - 1) / TILE;
    for (int t = 0; t < tiles; ++t) {
        int Acol = t * TILE + threadIdx.x;
        int Brow = t * TILE + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] = (row < A_rows && Acol < A_cols)
            ? A[row * A_cols + Acol]
            : 0;

        tileB[threadIdx.y][threadIdx.x] = (col < B_cols && Brow < A_cols)
            ? B[Brow * B_cols + col]
            : 0;

        __syncthreads();

        for (int k = 0; k < TILE; k++)
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];

        __syncthreads();
    }

    if (row < A_rows && col < B_cols)
        C[row * B_cols + col] = sum;
}

// ------------------------
// GPU METpHODS FOR Matrix<Tp>
// ------------------------

template<typename Tp>
void Matrix<Tp>::allocate_gpu() {
    cudaMalloc(&d_data, rows * cols * sizeof(Tp));
}

template<typename Tp>
void Matrix<Tp>::copy_to_gpu() {
    std::vector<Tp> flat(rows * cols);

    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            flat[i * cols + j] = matrix[i][j];

    cudaMemcpy(d_data, flat.data(), flat.size() * sizeof(Tp), cudaMemcpyHostToDevice);
}

template<typename Tp>
void Matrix<Tp>::copy_from_gpu() {
    std::vector<Tp> flat(rows * cols);

    cudaMemcpy(flat.data(), d_data, flat.size() * sizeof(Tp), cudaMemcpyDeviceToHost);

    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            matrix[i][j] = flat[i * cols + j];
}

template<typename Tp>
void Matrix<Tp>::free_gpu() {
    if (d_data) cudaFree(d_data);
}

template<typename Tp>
Matrix<Tp> Matrix<Tp>::operator+(const Matrix<Tp>& other) const {
    Matrix<Tp> result(rows, cols);
    result.allocate_gpu();

    int size = rows * cols;
    int block = 256;
    int grid = (size + block - 1) / block;

    elementwiseAddKernel<Tp><<<grid, block>>>(d_data, other.d_data, result.d_data, size);
    cudaDeviceSynchronize();

    result.copy_from_gpu();
    return result;
}

template<typename Tp>
Matrix<Tp> Matrix<Tp>::operator-(const Matrix<Tp>& other) const {
    Matrix<Tp> result(rows, cols);
    result.allocate_gpu();

    int size = rows * cols;
    int block = 256;
    int grid = (size + block - 1) / block;

    elementwiseSubKernel<Tp><<<grid, block>>>(d_data, other.d_data, result.d_data, size);
    cudaDeviceSynchronize();

    result.copy_from_gpu();
    return result;
}

template<typename Tp>
Matrix<Tp> Matrix<Tp>::operator*(const Matrix<Tp>& other) const {
    Matrix<Tp> result(rows, cols);
    result.allocate_gpu();

    int size = rows * cols;
    int block = 256;
    int grid = (size + block - 1) / block;

    elementwiseMulKernel<Tp><<<grid, block>>>(d_data, other.d_data, result.d_data, size);
    cudaDeviceSynchronize();

    result.copy_from_gpu();
    return result;
}

template<typename Tp>
Matrix<Tp> Matrix<Tp>::dot_gpu(const Matrix<Tp>& other) const {
    Matrix<Tp> result(rows, other.cols);
    result.allocate_gpu();

    dim3 block(TILE, TILE);
    dim3 grid((other.cols + TILE - 1) / TILE,
              (rows + TILE - 1) / TILE);

    matmulKernel<Tp><<<grid, block>>>(d_data, other.d_data, result.d_data, rows, cols, other.cols);
    cudaDeviceSynchronize();

    result.copy_from_gpu();
    return result;
}

template<typename Tp>
Matrix<Tp> Matrix<Tp>::relu_gpu() const {
    Matrix<Tp> result(rows, cols);
    result.allocate_gpu();

    int size = rows * cols;
    int block = 256;
    int grid = (size + block - 1) / block;

    reluKernel<<<grid, block>>>(d_data, result.d_data, size);
    cudaDeviceSynchronize();

    result.copy_from_gpu();
    return result;
}

template<typename Tp>
Matrix<Tp> Matrix<Tp>::tanh_gpu() const {
    Matrix<Tp> result(rows, cols);
    result.allocate_gpu();

    int size = rows * cols;
    int block = 256;
    int grid = (size + block - 1) / block;

    tanhKernel<<<grid, block>>>(d_data, result.d_data, size);
    cudaDeviceSynchronize();

    result.copy_from_gpu();
    return result;
}
template<typename Tp>
Matrix<Tp> Matrix<Tp>::softmax_gpu(Matrix<Tp>& mat) {
    int size = mat.rows * mat.cols;

    mat.allocate_gpu();
    mat.copy_to_gpu();

    Matrix<Tp> result(mat.rows, mat.cols);
    result.allocate_gpu();

    Tp* d_exp;
    cudaMalloc(&d_exp, size * sizeof(Tp));

    Tp* d_total;
    cudaMalloc(&d_total, sizeof(Tp));
    cudaMemset(d_total, 0, sizeof(Tp));

    int block = 256;
    int grid = (size + block - 1) / block;

    expKernel<Tp><<<grid, block>>>(mat.d_data, d_exp, size);
    sumKernel<Tp><<<grid, block>>>(d_exp, d_total, size);

    Tp total_val;
    cudaMemcpy(&total_val, d_total, sizeof(Tp), cudaMemcpyDeviceToHost);

    divKernel<Tp><<<grid, block>>>(d_exp, result.d_data, total_val, size);

    result.copy_from_gpu();

    cudaFree(d_exp);
    cudaFree(d_total);

    return result;
}

template<typename Tp>
void Matrix<Tp>::fill_gpu(Tp value) {
    int size = rows * cols;
    int block = 256;
    int grid = (size + block - 1) / block;

    fillKernel<Tp><<<grid, block>>>(d_data, value, size);
    cudaDeviceSynchronize();
}

template<typename Tp>
Matrix<Tp> Matrix<Tp>::scaled_gpu(Tp value) const {
    Matrix<Tp> result(rows, cols);
    result.allocate_gpu();

    int size = rows * cols;
    int block = 256;
    int grid = (size + block - 1) / block;

    cudaMemcpy(result.d_data, d_data, size * sizeof(Tp), cudaMemcpyDeviceToDevice);

    elementwiseScaleKernel<Tp><<<grid, block>>>(result.d_data, value, size);

    result.copy_from_gpu();
    return result;
}
// Instantiate templates
template class Matrix<float>;
template class Matrix<double>;
