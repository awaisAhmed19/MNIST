
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

struct ForwardCache {
    std::vector<std::unique_ptr<Tensor>> activations;
    std::vector<std::unique_ptr<Tensor>> zvals;
};

struct BackwardCache {
    std::vector<std::unique_ptr<Tensor>> dW;
    std::vector<std::unique_ptr<Tensor>> dB;
};

NeuralNetwork* Create(int input, int hidden, int output, float lr);
void Train_gpu(NeuralNetwork* net, Tensor* X, Tensor* Y);
void Train_batch_imgs(NeuralNetwork* net, std::vector<Filer::Img>& dataset, int batch_size);

std::unique_ptr<Tensor> predict_img(NeuralNetwork* net, Filer::Img& img);
float evaluate_accuracy(NeuralNetwork* net, std::vector<Filer::Img>& dataset, int n);
std::unique_ptr<Tensor> predict(NeuralNetwork* net, Tensor* input);

std::unique_ptr<Tensor> TaddBias(const Tensor& mat, const Tensor& bias);
std::unique_ptr<Tensor> stack_batch_inputs(const std::vector<Filer::Img>& dataset, int start,
                                           int batch_size);

std::unique_ptr<Tensor> stack_batch_labels(const std::vector<Filer::Img>& dataset, int start,
                                           int batch_size);
ForwardCache forward_pass_batch(NeuralNetwork* net, Tensor* X);
BackwardCache backward_pass_batch(NeuralNetwork* net, const ForwardCache& cache, Tensor* Y);
void update_params(NeuralNetwork* net, const BackwardCache& grads);
float cross_entropy_batch(const Tensor& predictions, const Tensor& targets);
float cross_entropy_loss(const Tensor& prediction, const Tensor& target);
void save(const NeuralNetwork* net, const std::string& filename);
NeuralNetwork* load(const std::string& filename);
void print(const NeuralNetwork* net);
void Train(NeuralNetwork* net, Tensor* X, Tensor* Y);
