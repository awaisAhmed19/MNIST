#include "neural_network.h"


void Train_gpu(NeuralNetwork* net, Matrix<float>& X, Matrix<float>& Y) {
    int L = net->layers.size() - 1;

    // Make sure ALL model params + inputs are on GPU
    X.allocate_gpu();
    X.copy_to_gpu();

    Y.allocate_gpu();
    Y.copy_to_gpu();

    for (int i = 0; i < L; ++i) {
        net->weights[i].allocate_gpu();
        net->weights[i].copy_to_gpu();

        net->biases[i].allocate_gpu();
        net->biases[i].copy_to_gpu();
    }

    std::vector<Matrix<float>> activation;
    std::vector<Matrix<float>> zvals;

    activation.push_back(X);
    Matrix<float> a = X;

    // FORWARD
    for (int i = 0; i < L; i++) {
        // sync 'a' to GPU before using
        a.allocate_gpu();
        a.copy_to_gpu();

        Matrix<float> z = net->weights[i].dot_gpu(a).add_gpu(net->biases[i]);
        Matrix<float> a_next = z.sigmoid_gpu();

        zvals.push_back(z);  // CPU copies for grad calc
        activation.push_back(a_next);

        a = a_next;
    }

    std::vector<Matrix<float>> grad(L);
    std::vector<Matrix<float>> deltaW(L);
    std::vector<Matrix<float>> deltaB(L);

    // Output error: A_L - Y
    activation[L].allocate_gpu();
    activation[L].copy_to_gpu();
    Y.allocate_gpu();
    Y.copy_to_gpu();

    Matrix<float> error = activation[L].sub_gpu(Y);

    zvals[L - 1].allocate_gpu();
    zvals[L - 1].copy_to_gpu();

    grad[L - 1] = error.mul_gpu(zvals[L - 1].sigmoid_prime_gpu());

    activation[L - 1].allocate_gpu();
    activation[L - 1].copy_to_gpu();

    deltaW[L - 1] =
        grad[L - 1].dot_gpu(activation[L - 1].transpose_gpu()).scaled_gpu(net->learningRate);
    deltaB[L - 1] = grad[L - 1].scaled_gpu(net->learningRate);

    // HIDDEN LAYERS
    for (int i = L - 2; i >= 0; i--) {
        net->weights[i + 1].allocate_gpu();
        net->weights[i + 1].copy_to_gpu();

        grad[i + 1].allocate_gpu();
        grad[i + 1].copy_to_gpu();

        Matrix<float> wT = net->weights[i + 1].transpose_gpu();
        Matrix<float> err = wT.dot_gpu(grad[i + 1]);

        zvals[i].allocate_gpu();
        zvals[i].copy_to_gpu();

        Matrix<float> g = err.mul_gpu(zvals[i].sigmoid_prime_gpu());
        grad[i] = g;

        activation[i].allocate_gpu();
        activation[i].copy_to_gpu();

        deltaW[i] = g.dot_gpu(activation[i].transpose_gpu()).scaled_gpu(net->learningRate);
        deltaB[i] = g.scaled_gpu(net->learningRate);
    }

    // UPDATE
    for (int i = 0; i < L; i++) {
        net->weights[i].allocate_gpu();
        net->weights[i].copy_to_gpu();

        deltaW[i].allocate_gpu();
        deltaW[i].copy_to_gpu();

        net->weights[i] = net->weights[i].add_gpu(deltaW[i]);

        net->biases[i].allocate_gpu();
        net->biases[i].copy_to_gpu();

        deltaB[i].allocate_gpu();
        deltaB[i].copy_to_gpu();

        net->biases[i] = net->biases[i].add_gpu(deltaB[i]);
    }
}
