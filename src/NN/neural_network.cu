#include "neural_network.h"
#include "../Tensor/tensor.h"

// GPU Forward Pass (batch)

ForwardCache forward_pass_gpu(NeuralNetwork* net, Tensor* X)
{
    int L = net->layers.size() - 1;

    ForwardCache cache;
    cache.activations.reserve(L + 1);
    cache.zvals.reserve(L);

    cache.activations.push_back(std::make_unique<Tensor>(*X));
    Tensor* a = cache.activations.back().get();

    for (int i = 0; i < L; i++) {

        // z = W[i] * a + b[i]
        auto z = TmatmulGPU(*net->weights[i], *a);
        auto z_bias = Tadd(*z, *net->biases[i]);   // bias add done on host (safe)

        cache.zvals.push_back(Tcopy(*z_bias));

        if (i == L - 1) {
            auto out = TSoftmaxCols(*z_bias);
            cache.activations.push_back(std::move(out));
        }
        else {
            auto out = Tcopy(*z_bias);
            TRelu(*out);
            cache.activations.push_back(std::move(out));
        }

        a = cache.activations.back().get();
    }

    return cache;
}

// GPU Backward Pass (batch)

BackwardCache backward_pass_gpu(
    NeuralNetwork* net,
    const ForwardCache& cache,
    Tensor* Y
) {
    int L = net->layers.size() - 1;
    BackwardCache grads;

    grads.dW.resize(L);
    grads.dB.resize(L);

    std::vector<std::unique_ptr<Tensor>> dZ(L);

    int batch = Y->cols;
    float scale = 1.0f / batch;

    // Output layer
    dZ[L - 1] = Tsub(*cache.activations[L], *Y);

    auto aPrevT = Ttranspose(*cache.activations[L - 1]);

    grads.dW[L - 1] =
        TmulScalar(*TmatmulGPU(*dZ[L - 1], *aPrevT), scale);

    grads.dB[L - 1] =
        TmulScalar(*TsumCols(*dZ[L - 1]), scale);

    // Hidden layers
    for (int i = L - 2; i >= 0; i--) {

        auto wT = Ttranspose(*net->weights[i + 1]);
        auto tmp = TmatmulGPU(*wT, *dZ[i + 1]);

        auto prime = Tcopy(*cache.zvals[i]);
        TReluPrime(*prime);

        dZ[i] = Tmul(*tmp, *prime);

        auto aT = Ttranspose(*cache.activations[i]);

        grads.dW[i] =
            TmulScalar(*TmatmulGPU(*dZ[i], *aT), scale);

        grads.dB[i] =
            TmulScalar(*TsumCols(*dZ[i]), scale);
    }

    return grads;
}

// GPU Training Wrapper

void Train_gpu(NeuralNetwork* net, Tensor* X, Tensor* Y)
{
    auto cache = forward_pass_gpu(net, X);
    auto grads = backward_pass_gpu(net, cache, Y);

    int L = net->layers.size() - 1;

    for (int i = 0; i < L; i++) {

        auto scaledW =
            TmulScalar(*grads.dW[i], net->learningRate);

        auto scaledB =
            TmulScalar(*grads.dB[i], net->learningRate);

        net->weights[i] = Tsub(*net->weights[i], *scaledW);
        net->biases[i]  = Tsub(*net->biases[i], *scaledB);
    }
}
