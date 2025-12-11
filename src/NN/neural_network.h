
#pragma once
#include <vector>

#include "../Filer.h"
#include "../Tensor/tensor.h"

struct NeuralNetwork {
    std::vector<int> layers;

    std::vector<std::unique_ptr<Tensor>> weights;
    std::vector<std::unique_ptr<Tensor>> biases;

    float learningRate;

    NeuralNetwork(const std::vector<int>& layerSize, float lr)
        : layers(layerSize), learningRate(lr) {
        int L = layers.size() - 1;

        for (int i = 0; i < L; i++) {
            int in = layers[i];
            int out = layers[i + 1];

            auto w = std::make_unique<Tensor>(out, in);
            auto b = std::make_unique<Tensor>(out, 1);

            TRandomize(*w, in);
            TRandomize(*b, in);

            weights.push_back(std::move(w));
            biases.push_back(std::move(b));
        }
    }
};

NeuralNetwork* Create(int input, int hidden, int output, float lr);
void Train(NeuralNetwork* net, Tensor* X, Tensor* Y);
void Train_gpu(NeuralNetwork* net, Tensor* X, Tensor* Y);
void Train_batch_imgs(NeuralNetwork* net, std::vector<Filer::Img>& dataset, int batch_size);

std::unique_ptr<Tensor> predict_img(NeuralNetwork* net, Filer::Img& img);
float evaluate_accuracy(NeuralNetwork* net, std::vector<Filer::Img>& dataset, int n);
std::unique_ptr<Tensor> predict(NeuralNetwork* net, Tensor* input);

float cross_entropy_batch(const Tensor& predictions, const Tensor& targets);
float cross_entropy_loss(const Tensor& prediction, const Tensor& target);
void save(const NeuralNetwork* net, const std::string& filename);
NeuralNetwork* load(const std::string& filename);
void print(const NeuralNetwork* net);
