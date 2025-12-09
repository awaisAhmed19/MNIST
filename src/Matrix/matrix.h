#pragma once
#include <cassert>
#include <iostream>
#include <random>
struct Tensor {
    int rows;
    int cols;

    float* h_data;  // CPU data
    float* d_data;  // GPU data

    bool dirty_host;
};

// allocation
Tensor* Tcreate(int r, int c);
void Tfree(Tensor* t);
void Tresize(Tensor* t, int r, int c);

// CPU ops (all return NEW tensors)
Tensor* Tcopy(Tensor* src);
Tensor* Tadd(Tensor* a, Tensor* b);
Tensor* Tsub(Tensor* a, Tensor* b);
Tensor* Tmul(Tensor* a, Tensor* b);
Tensor* Tdot(Tensor* a, Tensor* b);

Tensor* TmulScalar(Tensor* t, float s);
Tensor* TaddScalar(Tensor* t, float s);

// activations
Tensor* TSigmoid(Tensor* t);
Tensor* TSigmoidDeriv(Tensor* t);
Tensor* TRelu(Tensor* t);
Tensor* TReluDeriv(Tensor* t);
Tensor* TSoftmax(Tensor* t);

// utilities
void TRandomize(Tensor* t, float scale);
int TArgmax(Tensor* t);
float Tuni_dist(float l, float h);

// debugging
void TPrint(Tensor* t);
void TValidate(Tensor* t);
bool TCheckDimension(Tensor* t);
