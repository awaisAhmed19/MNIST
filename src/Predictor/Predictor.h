#pragma once

#include <iostream>
#include <string>
#include <vector>

#include "../Filer.h"
#include "../NN/neural_network.h"
#include "../Tensor/tensor.h"
#include "nn.h"

extern Filer filer;

extern int g_lastPrediction;

// ---------------- Visualization ----------------

// Timeline storage (DEFINED IN .cpp)
bool init_predictor();
void predict_on_save(const std::string& path);
void print(const std::string& file);
