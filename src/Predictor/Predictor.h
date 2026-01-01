#pragma once

#include <iomanip>
#include <iostream>
#include <string>

#include "../Filer.h"
#include "../NN/neural_network.h"
#include "../Tensor/tensor.h"
extern Filer filer;

void print(const std::string& file);
void predict_on_save(const std::string& pred_in);
