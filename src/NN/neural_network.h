#pragma once
#include "../Filer.h"
#include "../Matrix/matrix.h"

struct NeuralNetwork {
    int input;
    int hidden;
    int output;
    double learningRate;
    Matrix hiddenWeights;
    Matrix outputWeights;
};

NeuralNetwork* Create(int input, int hidden, int output, double lr);
void Train(NeuralNetwork* net, Matrix& X, Matrix& Y);
void Train_batch_imgs(NeuralNetwork* net, std::vector<Filer::Img>& dataset, int batch_size);
Matrix predict_img(const NeuralNetwork* net, const Filer::Img& img);
double evaluate_accuracy(const NeuralNetwork* net, const std::vector<Filer::Img>& dataset, int n);
Matrix predict(const NeuralNetwork* net, const Matrix& input);
void save(const NeuralNetwork* net, const std::string& filename);
NeuralNetwork* load(const std::string& filename);
void print(const NeuralNetwork* net);
