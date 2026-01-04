#include "neural_network.h"
#include "../Tensor/tensor.h"

void Train_gpu(NeuralNetwork* net, Tensor* X, Tensor* Y)
{
    int L = net->layers.size() - 1;

    TtoDevice(X);
    TtoDevice(Y);

    for (int i = 0; i < L; ++i) {
        TtoDevice(net->weights[i].get());
        TtoDevice(net->biases[i].get());
    }

    std::vector<Tensor*> activation(L + 1);
    std::vector<Tensor*> zvals(L);

    activation[0] = TcopyGPU(X);

    // ---------- FORWARD ----------
    for (int i = 0; i < L; i++) {

        Tensor* z      = TmatmulGPU(activation[i], net->weights[i].get());
        Tensor* z_bias = TaddGPU(z, net->biases[i].get());

        zvals[i] = TcopyGPU(z_bias);

        Tensor* act = TSigmoidGPU(zvals[i]);
        activation[i + 1] = act;
    }

    // ---------- BACKWARD ----------
    std::vector<Tensor*> grad(L);
    std::vector<Tensor*> deltaW(L);
    std::vector<Tensor*> deltaB(L);

    Tensor* error = TsubGPU(activation[L], Y);

    Tensor* sp = TSigmoidPrimeGPU(zvals[L - 1]);
    grad[L - 1] = TmulGPU(error, sp);      // elementwise

    Tensor* a_prev_T = TtransposeGPU(activation[L - 1]);
    deltaW[L - 1] = TscaleGPU(
        TmatmulGPU(grad[L - 1], a_prev_T),
        net->learningRate
    );

    deltaB[L - 1] = TscaleGPU(grad[L - 1], net->learningRate);

    for (int i = L - 2; i >= 0; --i) {

        Tensor* wT = TtransposeGPU(net->weights[i + 1].get());
        Tensor* err = TmatmulGPU(wT, grad[i + 1]);

        Tensor* sp_i = TSigmoidPrimeGPU(zvals[i]);
        grad[i] = TmulGPU(err, sp_i);

        Tensor* aT = TtransposeGPU(activation[i]);
        deltaW[i] = TscaleGPU(
            TmatmulGPU(grad[i], aT),
            net->learningRate
        );

        deltaB[i] = TscaleGPU(grad[i], net->learningRate);
    }

    // ---------- UPDATE ----------
    for (int i = 0; i < L; ++i) {

        Tensor* w_new = TaddGPU(net->weights[i].get(), deltaW[i]);
        Tensor* b_new = TaddGPU(net->biases[i].get(), deltaB[i]);

        TtoHost(w_new);
        TtoHost(b_new);

        memcpy(net->weights[i]->h_data, w_new->h_data,
               sizeof(float) * Tsize(net->weights[i].get()));

        memcpy(net->biases[i]->h_data, b_new->h_data,
               sizeof(float) * Tsize(net->biases[i].get()));

        net->weights[i]->dirty_device = true;
        net->biases[i]->dirty_device  = true;
    }
}
