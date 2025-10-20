#pragma once
#include "../Matrix/matrix.h"

class Matrix;

typedef struct {
    int input;
    int hidden;
    int output;

    double learningRate;
    Matrix hiddenWeights;
    Matrix outputWeight;

} NeuralNetwork;

NeuralNetwork* Create(int input, int hidden, int output, double lr);
void Train(NeuralNetwork* net, Matrix& input, Matrix& output);
