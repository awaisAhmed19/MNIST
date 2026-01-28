#pragma once

#ifdef USE_CUDA
#include <cuda_runtime_api.h>

#define CUDA_CHECK(err)                                                                \
    if (err != cudaSuccess) {                                                          \
        printf("CUDA ERROR %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1);                                                                       \
    }

#endif  // USE_CUDA

#include <cassert>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>

struct Tensor {
    int rows, cols;

    float* h_data = nullptr;
    float* d_data = nullptr;

    bool dirty_host = false;    // GPU wrote -> host stale
    bool dirty_device = false;  // CPU wrote -> device stale

    Tensor(int r, int c) : rows(r), cols(c) {
        if (r <= 0 || c <= 0) throw std::runtime_error("Invalid tensor size");

        int size = r * c;
        h_data = new float[size]();

#ifdef USE_CUDA
        cudaMalloc((void**)&d_data, size * sizeof(float));
        cudaMemset(d_data, 0, size * sizeof(float));
#endif
    }

    Tensor() : rows(0), cols(0), h_data(nullptr), d_data(nullptr) {}

    Tensor(const Tensor& other) : rows(other.rows), cols(other.cols) {
        int size = rows * cols;
        h_data = new float[size];
        memcpy(h_data, other.h_data, size * sizeof(float));

#ifdef USE_CUDA
        if (other.d_data) {
            cudaMalloc((void**)&d_data, size * sizeof(float));
            cudaMemcpy(d_data, other.d_data, size * sizeof(float), cudaMemcpyDeviceToDevice);
        } else
            d_data = nullptr;
#else
        d_data = nullptr;
#endif

        dirty_host = false;
        dirty_device = false;
    }

    Tensor(Tensor&& other) noexcept
        : rows(other.rows),
          cols(other.cols),
          h_data(other.h_data),
          d_data(other.d_data),
          dirty_host(other.dirty_host),
          dirty_device(other.dirty_device) {
        other.rows = 0;
        other.cols = 0;
        other.h_data = nullptr;
        other.d_data = nullptr;
    }

    Tensor& operator=(const Tensor& other) {
        if (this == &other) return *this;

        if (h_data) delete[] h_data;
#ifdef USE_CUDA
        if (d_data) cudaFree(d_data);
#endif

        rows = other.rows;
        cols = other.cols;
        int size = rows * cols;

        h_data = new float[size];
        memcpy(h_data, other.h_data, size * sizeof(float));

#ifdef USE_CUDA
        if (other.d_data) {
            cudaMalloc((void**)&d_data, size * sizeof(float));
            cudaMemcpy(d_data, other.d_data, size * sizeof(float), cudaMemcpyDeviceToDevice);
        } else
            d_data = nullptr;
#else
        d_data = nullptr;
#endif

        dirty_host = false;
        dirty_device = false;
        return *this;
    }

    Tensor& operator=(Tensor&& other) noexcept {
        if (this == &other) return *this;

        if (h_data) delete[] h_data;
#ifdef USE_CUDA
        if (d_data) cudaFree(d_data);
#endif

        rows = other.rows;
        cols = other.cols;
        h_data = other.h_data;
        d_data = other.d_data;
        dirty_host = other.dirty_host;
        dirty_device = other.dirty_device;

        other.rows = 0;
        other.cols = 0;
        other.h_data = nullptr;
        other.d_data = nullptr;

        return *this;
    }

    ~Tensor() {
        if (h_data) delete[] h_data;
#ifdef USE_CUDA
        if (d_data) cudaFree(d_data);
#endif
    }

    inline int size() const { return rows * cols; }
};

// ---------------- allocation ----------------

Tensor* Tcreate(int r, int c);
void Tfree(Tensor*& t);
void Tresize(Tensor* t, int r, int c);

std::unique_ptr<Tensor> Tonehot(int label);

// sync helpers (REFERENCE now)
void TtoDevice(Tensor& t);
void TtoHost(Tensor& t);

// ---------------- CPU ops ----------------

std::unique_ptr<Tensor> Tcopy(const Tensor& src);
std::unique_ptr<Tensor> Tadd(const Tensor& a, const Tensor& b);
std::unique_ptr<Tensor> Tsub(const Tensor& a, const Tensor& b);
std::unique_ptr<Tensor> Tmul(const Tensor& a, const Tensor& b);
std::unique_ptr<Tensor> Tmatmul(const Tensor& A, const Tensor& B);

std::unique_ptr<Tensor> Tmatmul_R(const Tensor& a, const Tensor& b, int layer);

std::unique_ptr<Tensor> TmulScalar(const Tensor& in, float s);
std::unique_ptr<Tensor> TaddScalar(const Tensor& in, float s);
std::unique_ptr<Tensor> TsumCols(const Tensor& t);

std::unique_ptr<Tensor> TaddBias(const Tensor& mat, const Tensor& bias);
std::unique_ptr<Tensor> TaddBias_R(const Tensor& mat, const Tensor& bias, int layer);
// ---------------- CUDA ops (NEW API) ----------------
#ifdef USE_CUDA

void TupdateGPU(Tensor& W, const Tensor& dW, float lr);
std::unique_ptr<Tensor> TaddGPU(Tensor& A, Tensor& B);
std::unique_ptr<Tensor> TsubGPU(Tensor& A, Tensor& B);
std::unique_ptr<Tensor> TmulGPU(Tensor& A, Tensor& B);
std::unique_ptr<Tensor> TmatmulGPU(Tensor& A, Tensor& B);
std::unique_ptr<Tensor> TscaleGPU(Tensor& A, float s);
std::unique_ptr<Tensor> TSigmoidGPU(Tensor& A);
std::unique_ptr<Tensor> TSigmoidPrimeGPU(Tensor& A);
void TReluGPU(Tensor& A);
std::unique_ptr<Tensor> TReluPrimeGPU(Tensor& A);
std::unique_ptr<Tensor> TSoftmaxRowsGPU(Tensor& A);
std::unique_ptr<Tensor> TSoftmaxColsGPU(Tensor& A);
std::unique_ptr<Tensor> TtransposeGPU(Tensor& A);

#endif  // USE_CUDA

// ---------------- activations ----------------

std::unique_ptr<Tensor> TSigmoid(const Tensor& src);
std::unique_ptr<Tensor> TSigmoidPrime(const Tensor& src);
void TRelu(Tensor& t);
void TReluPrime(Tensor& t);
std::unique_ptr<Tensor> TSoftmaxRows(const Tensor& src);
std::unique_ptr<Tensor> TSoftmaxCols(const Tensor& src);

// ---------------- utils ----------------

void TRandomize(Tensor& t, float fan_in);
int TArgmax(const Tensor& t);
void TPrint(const Tensor& t);

std::unique_ptr<Tensor> Ttranspose(const Tensor& t);
std::unique_ptr<Tensor> Tflatten(const Tensor& t);
std::unique_ptr<Tensor> Ttanh(const Tensor& t);

float Tuni_dist_std(float l, float h);
void TValidate(Tensor* t);
bool TCheckDimension(Tensor* t);
