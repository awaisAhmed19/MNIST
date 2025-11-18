#pragma once
#include <vector>

#include "../Filer.h"
#include "../Matrix/matrix.h"

struct NeuralNetwork {
    std::vector<int> layers;
    std::vector<Matrix<double>> weights;
    std::vector<Matrix<double>> biases;
    double learningRate;

    NeuralNetwork(const std::vector<int>& layerSize, double lr)
        : layers(layerSize), learningRate(lr) {
        for (int i = 0; i < layers.size() - 1; i++) {
            Matrix<double> w(layers[i + 1], layers[i]);
            Matrix<double> b(layers[i + 1], 1);
            randomize(w);
            randomize(b);
            weights.push_back(w);
            biases.push_back(b);
        }
    }

   private:
    void randomize(Matrix<double>& mat) {
        for (int i = 0; i < (int)mat.row(); ++i)
            for (int j = 0; j < (int)mat.col(); ++j)
                mat.matrix[i][j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;  // range [-1, 1]
    }
};

NeuralNetwork* Create(int input, int hidden, int output, double lr);
void Train(NeuralNetwork* net, Matrix<double>& X, Matrix<double>& Y);
void Train_batch_imgs(NeuralNetwork* net, std::vector<Filer::Img>& dataset, int batch_size);
Matrix<double> predict_img(NeuralNetwork* net, Filer::Img& img);
double evaluate_accuracy(NeuralNetwork* net, std::vector<Filer::Img>& dataset, int n);
Matrix<double> predict(NeuralNetwork* net, Matrix<double>& input);
void save(const NeuralNetwork* net, const std::string& filename);
NeuralNetwork* load(const std::string& filename);
void print(const NeuralNetwork* net);
