#pragma once

#include <iomanip>
#include <iostream>
#include <string>

#include "../Filer.h"
#include "../NN/neural_network.h"
#include "../Tensor/tensor.h"
extern Filer filer;

enum class VizOp { Copy, MatMul, AddBias, ReLU, Softmax };

struct VizEvent {
    VizOp op;
    const Tensor* A;    // input
    const Tensor* B;    // optional (weights/bias)
    const Tensor* Out;  // output
};

using VizCallback = void (*)(const VizEvent&);

extern VizCallback g_viz;
void print(const std::string& file);
void predict_on_save(const std::string& pred_in);
