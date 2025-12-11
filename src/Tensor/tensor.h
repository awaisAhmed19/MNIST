#pragma once
#ifdef USE_CUDA

#include <cuda_runtime_api.h>
#endif  // USE_CUDA

#include <cassert>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#define CUDA_CHECK(err)                                                                \
    if (err != cudaSuccess) {                                                          \
        printf("CUDA ERROR %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1);                                                                       \
    }
#include <cstring>
#include <iostream>
#include <stdexcept>

struct Tensor {
    int rows;
    int cols;

    float* h_data;  // CPU memory (host)
    float* d_data;  // GPU memory (device)

    bool dirty_host;    // host is out-of-sync
    bool dirty_device;  // device is out-of-sync

    Tensor(int r, int c)
        : rows(r),
          cols(c),
          h_data(nullptr),
          d_data(nullptr),
          dirty_host(false),
          dirty_device(false) {
        if (r <= 0 || c <= 0) throw std::runtime_error("Invalid tensor size");

        int size = r * c;

        // CPU allocation
        h_data = new float[size]();

#ifdef USE_CUDA
        cudaMalloc(&d_data, size * sizeof(float));
        cudaMemset(d_data, 0, size * sizeof(float));
#endif
    }

    Tensor()
        : rows(0),
          cols(0),
          h_data(nullptr),
          d_data(nullptr),
          dirty_host(false),
          dirty_device(false) {}

    Tensor(const Tensor& other)
        : rows(other.rows), cols(other.cols), dirty_host(false), dirty_device(false) {
        int size = rows * cols;

        // Copy host
        h_data = new float[size];
        std::memcpy(h_data, other.h_data, size * sizeof(float));

#ifdef USE_CUDA
        if (other.d_data) {
            cudaMalloc(&d_data, size * sizeof(float));
            cudaMemcpy(d_data, other.d_data, size * sizeof(float), cudaMemcpyDeviceToDevice);
        } else {
            d_data = nullptr;
        }
#else
        d_data = nullptr;
#endif
    }

    Tensor(Tensor&& other) noexcept
        : rows(other.rows),
          cols(other.cols),
          h_data(other.h_data),
          d_data(other.d_data),
          dirty_host(other.dirty_host),
          dirty_device(other.dirty_device) {
        other.h_data = nullptr;
        other.d_data = nullptr;
        other.rows = 0;
        other.cols = 0;
    }

    Tensor& operator=(const Tensor& other) {
        if (this == &other) return *this;

        // Free old memory
        if (h_data) delete[] h_data;
#ifdef USE_CUDA
        if (d_data) cudaFree(d_data);
#endif

        // Copy new info
        rows = other.rows;
        cols = other.cols;
        int size = rows * cols;

        h_data = new float[size];
        std::memcpy(h_data, other.h_data, size * sizeof(float));

#ifdef USE_CUDA
        if (other.d_data) {
            cudaMalloc(&d_data, size * sizeof(float));
            cudaMemcpy(d_data, other.d_data, size * sizeof(float), cudaMemcpyDeviceToDevice);
        } else {
            d_data = nullptr;
        }
#else
        d_data = nullptr;
#endif

        dirty_host = false;
        dirty_device = false;

        return *this;
    }

    // ------------------------------------
    // Move Assignment Operator
    // ------------------------------------
    Tensor& operator=(Tensor&& other) noexcept {
        if (this == &other) return *this;

        // Free old memory
        if (h_data) delete[] h_data;
#ifdef USE_CUDA
        if (d_data) cudaFree(d_data);
#endif

        // Transfer ownership
        rows = other.rows;
        cols = other.cols;
        h_data = other.h_data;
        d_data = other.d_data;
        dirty_host = other.dirty_host;
        dirty_device = other.dirty_device;

        other.h_data = nullptr;
        other.d_data = nullptr;
        other.rows = 0;
        other.cols = 0;

        return *this;
    }

    ~Tensor() {
#ifdef USE_CUDA
        if (d_data) cudaFree(d_data);
#endif
        if (h_data) delete[] h_data;
    }

    inline int size() const { return rows * cols; }
};

// allocation
Tensor* Tcreate(int r, int c);

void Tfree(Tensor*& t);
void Tresize(Tensor* t, int r, int c);

std::unique_ptr<Tensor> Tonehot(int label);
void TtoDevice(Tensor* t);
void TtoHost(Tensor* t);
// CPU ops (all return NEW tensors)
std::unique_ptr<Tensor> Tcopy(const Tensor& src);
std::unique_ptr<Tensor> Tadd(const Tensor& a, const Tensor& b);
std::unique_ptr<Tensor> Tsub(const Tensor& a, const Tensor& b);
std::unique_ptr<Tensor> Tmul(const Tensor& a, const Tensor& b);
std::unique_ptr<Tensor> Tmatmul(const Tensor& A, const Tensor& B);
std::unique_ptr<Tensor> TmulScalar(const Tensor& in, float s);
std::unique_ptr<Tensor> TaddScalar(const Tensor& in, float s);

// activations

std::unique_ptr<Tensor> TSigmoid(const Tensor& src);
std::unique_ptr<Tensor> TSigmoidPrime(const Tensor& src);
void TRelu(Tensor& t);
void TReluPrime(Tensor& t);
void TSoftmaxRows(Tensor& t);
void TRandomize(Tensor& t, float fan_in);
void TPrint(const Tensor& t);
int TArgmax(const Tensor& t);

std::unique_ptr<Tensor> Ttranspose(const Tensor& t);
std::unique_ptr<Tensor> Tflatten(const Tensor& t);
std::unique_ptr<Tensor> Ttanh(const Tensor& t);
void TValidate(Tensor* t);
bool TCheckDimension(Tensor* t);
