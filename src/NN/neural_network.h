#pragma once
#include <vector>

#include "../Filer.h"
#include "../Matrix/matrix.h"

struct NeuralNetwork {
    std::vector<int> layers;
    std::vector<Matrix<float>> weights;
    std::vector<Matrix<float>> biases;
    float learningRate;

    NeuralNetwork(const std::vector<int>& layerSize, float lr)
        : layers(layerSize), learningRate(lr) {
        for (int i = 0; i < layers.size() - 1; i++) {
            Matrix<float> w(layers[i + 1], layers[i]);
            Matrix<float> b(layers[i + 1], 1);
            randomize(w);
            randomize(b);
            weights.push_back(w);
            biases.push_back(b);
        }
    }

   private:
    void randomize(Matrix<float>& mat) {
        for (int i = 0; i < (int)mat.row(); ++i)
            for (int j = 0; j < (int)mat.col(); ++j)
                mat.matrix[i][j] = ((float)rand() / RAND_MAX) * 2.0 - 1.0;  // range [-1, 1]
    }
};

NeuralNetwork* Create(int input, int hidden, int output, float lr);
void Train(NeuralNetwork* net, Matrix<float>& X, Matrix<float>& Y);
void Train_batch_imgs(NeuralNetwork* net, std::vector<Filer::Img>& dataset, int batch_size);
Matrix<float> predict_img(NeuralNetwork* net, Filer::Img& img);
float evaluate_accuracy(NeuralNetwork* net, std::vector<Filer::Img>& dataset, int n);
Matrix<float> predict(NeuralNetwork* net, Matrix<float>& input);
void save(const NeuralNetwork* net, const std::string& filename);
NeuralNetwork* load(const std::string& filename);
void print(const NeuralNetwork* net);
