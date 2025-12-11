#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

#include "neural_network.h"

struct ForwardCache {
    std::vector<std::unique_ptr<Tensor>> activations;
    std::vector<std::unique_ptr<Tensor>> zvals;
};

struct BackwardCache {
    std::vector<std::unique_ptr<Tensor>> dW;
    std::vector<std::unique_ptr<Tensor>> dB;
};

NeuralNetwork* Create(int input, int hidden, int output, float lr) {
    std::vector<int> layers = {input, hidden, output};
    return new NeuralNetwork(layers, lr);
}

/*
Tmatmul
Tadd
TSigmoid

Backprop
Output layer:
error = Y - aL
grad[L] = error ⊙ sigmoid'(aL)
deltaW[L] = grad[L] * a(L-1)^T
deltaB[L] = grad[L]

Hidden layers:
grad[i] = (W[i+1]^T * grad[i+1]) ⊙ sigmoid'(a[i])
deltaW[i] = grad[i] * a[i-1]^T
deltaB[i] = grad[i]

update
  W += -lr * dW
B += -lr * dB
*/

ForwardCache forward_pass(NeuralNetwork* net, Tensor* X) {
    int L = net->layers.size() - 1;
    ForwardCache cache;

    cache.activations.push_back(std::make_unique<Tensor>(*X));
    Tensor* a = cache.activations.back().get();

    for (int i = 0; i < L; i++) {
        auto z = Tadd(*Tmatmul(*net->weights[i], *a), *net->biases[i]);
        cache.zvals.push_back(Tcopy(*z));

        if (i == L - 1) {
            // OUTPUT LAYER = SOFTMAX
            auto a_next = Tcopy(*z);
            TSoftmaxRows(*a_next);
            cache.activations.push_back(std::move(a_next));
        } else {
            // HIDDEN LAYERS = SIGMOID
            auto a_next = TSigmoid(*Tcopy(*z));
            cache.activations.push_back(std::move(a_next));
        }

        a = cache.activations.back().get();
    }

    return cache;
}

BackwardCache backward_pass(NeuralNetwork* net, const ForwardCache& cache, Tensor* Y) {
    int L = net->layers.size() - 1;
    BackwardCache grads;

    grads.dW.resize(L);
    grads.dB.resize(L);

    std::vector<std::unique_ptr<Tensor>> dZ(L);

    // ---- OUTPUT LAYER (softmax + CE) ----
    dZ[L - 1] = Tsub(*cache.activations[L], *Y);

    auto a_prev_T = Ttranspose(*cache.activations[L - 1]);
    grads.dW[L - 1] = Tmatmul(*dZ[L - 1], *a_prev_T);
    grads.dB[L - 1] = Tcopy(*dZ[L - 1]);

    // ---- HIDDEN LAYERS ----
    for (int i = L - 2; i >= 0; i--) {
        auto wT = Ttranspose(*net->weights[i + 1]);
        auto tmp = Tmatmul(*wT, *dZ[i + 1]);

        auto actPrime = TSigmoidPrime(*Tcopy(*cache.zvals[i]));
        dZ[i] = Tmul(*tmp, *actPrime);

        auto aT = Ttranspose(*cache.activations[i]);

        grads.dW[i] = Tmatmul(*dZ[i], *aT);
        grads.dB[i] = Tcopy(*dZ[i]);
    }

    return grads;
}

void update_params(NeuralNetwork* net, const BackwardCache& grads) {
    int L = net->layers.size() - 1;

    for (int i = 0; i < L; i++) {
        auto scaled_dW = TmulScalar(*grads.dW[i], net->learningRate);
        auto scaled_dB = TmulScalar(*grads.dB[i], net->learningRate);

        net->weights[i] = Tsub(*net->weights[i], *scaled_dW);
        net->biases[i] = Tsub(*net->biases[i], *scaled_dB);
    }
}

void Train(NeuralNetwork* net, Tensor* X, Tensor* Y) {
    auto cache = forward_pass(net, X);
    auto grads = backward_pass(net, cache, Y);
    update_params(net, grads);
}

void Train_batch_imgs(NeuralNetwork* net, std::vector<Filer::Img>& dataset, int batch_size) {
    int limit = std::min(batch_size, (int)dataset.size());

    for (int i = 0; i < limit; i++) {
        Filer::Img& curr = dataset[i];

        auto Image_vec = Tflatten(*curr.img_data);  // unique_ptr<Tensor>
        auto output = Tonehot(curr.label);          // unique_ptr<Tensor>

#ifdef USE_CUDA
        Train_gpu(net, Image_vec.get(), output.get());
#else
        Train(net, Image_vec.get(), output.get());
#endif
        // NO FREE — unique_ptr handles it.
    }
}

// TODO: turn this into gpu code as well ??

// loss = - sum_i target_i * log(pred_i + eps)
// returns scalar loss for the single sample
float cross_entropy_loss(const Tensor& prediction, const Tensor& target) {
    if (prediction.rows != target.rows || prediction.cols != target.cols) {
        throw std::runtime_error("cross_entropy_loss: shape mismatch");
    }

    const float eps = 1e-12f;
    float loss = 0.0f;

    int size = prediction.rows * prediction.cols;
    for (int k = 0; k < size; ++k) {
        float y = target.h_data[k];
        // only accumulate where target is nonzero (one-hot) but this also works with soft targets
        if (y != 0.0f) {
            float p = prediction.h_data[k];
            loss -= y * std::log(p + eps);
        }
    }

    return loss;
}
float cross_entropy_batch(const Tensor& predictions, const Tensor& targets) {
    // predictions: (batch x 10)
    // targets:     (batch x 10)

    const float eps = 1e-12f;
    float loss = 0.0f;

    for (int b = 0; b < predictions.rows; b++) {
        for (int i = 0; i < predictions.cols; i++) {
            float y = targets.h_data[b * targets.cols + i];
            float p = predictions.h_data[b * predictions.cols + i];
            if (y > 0.0f) loss -= std::log(p + eps);
        }
    }

    return loss / predictions.rows;  // mean loss
}
std::unique_ptr<Tensor> predict_img(NeuralNetwork* net, Filer::Img& img) {
    auto Image_vec = Tflatten(*img.img_data);
    return predict(net, Image_vec.get());  // predict returns unique_ptr<Tensor>
}

float evaluate_accuracy(NeuralNetwork* net, std::vector<Filer::Img>& dataset, int n) {
    int correct = 0;

    for (int i = 0; i < n; i++) {
        auto prediction = predict_img(net, dataset[i]);  // unique_ptr<Tensor>
        int predicted_class = TArgmax(*prediction);

        if (predicted_class == dataset[i].label) correct++;
    }

    return (float)correct / n;
}

std::unique_ptr<Tensor> predict(NeuralNetwork* net, Tensor* input) {
    Tensor* a = input;
    int L = net->layers.size() - 1;

    std::unique_ptr<Tensor> out;

    for (int i = 0; i < L; i++) {
        auto z = Tmatmul(*net->weights[i], *a);  // unique_ptr<Tensor>
        auto z2 = Tadd(*z, *net->biases[i]);     // unique_ptr<Tensor>

        if (i < L - 1) {
            TSigmoid(*z2);  // in-place
        } else {
            TSoftmaxRows(*z2);  // in-place
        }

        out = std::move(z2);  // store output
        a = out.get();        // next iteration uses raw pointer
    }

    return out;  // unique_ptr moves out cleanly
}
void save(const NeuralNetwork* net, const std::string& dir_name) {
    namespace fs = std::filesystem;
    fs::path dir = dir_name;

    try {
        fs::create_directories(dir);

        // Save architecture
        std::ofstream desc(dir / "descriptor.txt");
        if (!desc) {
            std::cerr << "Error: failed to open descriptor file.\n";
            return;
        }

        desc << net->layers.size() << "\n";
        for (int size : net->layers) desc << size << "\n";

        desc << net->learningRate << "\n";

        // Save each tensor
        for (int i = 0; i < net->layers.size() - 1; i++) {
            std::string wFile = "weights_" + std::to_string(i) + ".csv";
            std::string bFile = "biases_" + std::to_string(i) + ".csv";

            Filer::save_tensor(net->weights[i].get(), (dir / wFile).string());
            Filer::save_tensor(net->biases[i].get(), (dir / bFile).string());
        }

        std::cout << "Network saved successfully in: " << dir << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Save error: " << e.what() << "\n";
    }
}

NeuralNetwork* load(const std::string& dir_name) {
    namespace fs = std::filesystem;
    fs::path dir = dir_name;

    if (!fs::exists(dir)) {
        std::cerr << "Directory doesn’t exist.\n";
        return nullptr;
    }

    try {
        // -----------------------
        // Load descriptor
        // -----------------------
        std::ifstream desc(dir / "descriptor.txt");
        if (!desc) {
            std::cerr << "Descriptor missing.\n";
            return nullptr;
        }

        int L;
        desc >> L;

        std::vector<int> layers(L);
        for (int i = 0; i < L; i++) desc >> layers[i];

        float lr;
        desc >> lr;

        // Create network object
        auto* net = new NeuralNetwork(layers, lr);

        // -----------------------
        // Load weights + biases
        // -----------------------
        for (int i = 0; i < L - 1; i++) {
            std::string wFile = "weights_" + std::to_string(i) + ".csv";
            std::string bFile = "biases_" + std::to_string(i) + ".csv";

            // Load raw tensor

            auto w_raw = Filer::load_tensor((dir / wFile).string());
            auto b_raw = Filer::load_tensor((dir / bFile).string());

            if (!w_raw || !b_raw) {
                std::cerr << "Failed loading tensor for layer " << i << "\n";
                delete net;
                return nullptr;
            }

            // Replace unique_ptr contents

            net->weights[i] = std::move(w_raw);
            net->biases[i] = std::move(b_raw);
        }

        std::cout << "Loaded network from: " << dir_name << "\n";
        return net;
    } catch (const std::exception& e) {
        std::cerr << "Load error: " << e.what() << "\n";
        return nullptr;
    }
}
/*
void print(const NeuralNetwork* net) {
    std::cout << "\n===== Neural Network =====\n";

    std::cout << "Layers: ";
    for (size_t i = 0; i < net->layers.size(); i++) {
        std::cout << net->layers[i];
        if (i < net->layers.size() - 1) std::cout << " -> ";
    }
    std::cout << "\n";

    std::cout << "Learning Rate: " << net->learningRate << "\n\n";

    int L = net->layers.size() - 1;

    for (int i = 0; i < L; i++) {
        std::cout << "=== Layer " << i << " → " << (i + 1) << " ===\n";
        std::cout << "Weights (" << net->weights[i].row() << "x" << net->weights[i].col() << ")\n";
        net->weights[i].print();

        std::cout << "Biases (" << net->biases[i].row() << "x" << net->biases[i].col() << ")\n";
        net->biases[i].print();

        std::cout << "\n";
    }

    std::cout << "==========================\n";
}*/
