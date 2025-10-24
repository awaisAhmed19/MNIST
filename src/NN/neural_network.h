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

    NeuralNetwork(int inputSize, int hiddenSize, int outputSize, double lr)
        : input(inputSize),
          hidden(hiddenSize),
          output(outputSize),
          learningRate(lr),
          hiddenWeights(hiddenSize, inputSize),
          outputWeights(outputSize, hiddenSize) {
        // initialize with small random values
        randomize(hiddenWeights);
        randomize(outputWeights);
    }

   private:
    void randomize(Matrix& mat) {
        for (int i = 0; i < mat.m_rows; ++i)
            for (int j = 0; j < mat.m_cols; ++j)
                mat.m_samples[i][j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;  // range [-1, 1]
    }
};

NeuralNetwork* Create(int input, int hidden, int output, double lr);
void Train(NeuralNetwork* net, Matrix& X, Matrix& Y);
void Train_batch_imgs(NeuralNetwork* net, std::vector<Filer::Img>& dataset, int batch_size);
Matrix predict_img(NeuralNetwork* net, Filer::Img& img);
double evaluate_accuracy(NeuralNetwork* net, std::vector<Filer::Img>& dataset, int n);
Matrix predict(NeuralNetwork* net, Matrix& input);
void save(const NeuralNetwork* net, const std::string& filename);
NeuralNetwork* load(const std::string& filename);
void print(const NeuralNetwork* net);
