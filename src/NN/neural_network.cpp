#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

#include "neural_network.h"

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
void Train(NeuralNetwork* net, Tensor* X, Tensor* Y) {
    int L = net->layers.size() - 1;

    // ----------------------------
    // FORWARD PASS
    // ----------------------------
    std::vector<std::unique_ptr<Tensor>> activations;
    std::vector<std::unique_ptr<Tensor>> zvals;

    activations.push_back(std::make_unique<Tensor>(*X));  // a0 = X
    Tensor* a = activations.back().get();

    for (int i = 0; i < L; i++) {
        auto z = Tadd(*Tmatmul(*net->weights[i], *a),  // W*a
                      *net->biases[i]                  // + b
        );

        zvals.push_back(Tcopy(*z));  // store raw z for backprop

        auto a_next = TSigmoid(*Tcopy(*z));  // activation = sigmoid(z)

        activations.push_back(std::move(a_next));
        a = activations.back().get();
    }

    // ----------------------------
    // BACKWARD PASS
    // ----------------------------
    std::vector<std::unique_ptr<Tensor>> grad(L);
    std::vector<std::unique_ptr<Tensor>> deltaW(L);
    std::vector<std::unique_ptr<Tensor>> deltaB(L);

    // ---- Output layer ----
    auto error = Tsub(*Y, *activations[L]);  // error = y - a_L
    auto actPrime = TSigmoidPrime(*Tcopy(*zvals[L - 1]));

    grad[L - 1] = Tmul(*error, *actPrime);

    auto a_prev_T = Ttranspose(*activations[L - 1]);
    deltaW[L - 1] = TmulScalar(*Tmatmul(*grad[L - 1], *a_prev_T), net->learningRate);
    deltaB[L - 1] = TmulScalar(*grad[L - 1], net->learningRate);

    // ---- Hidden layers ----
    for (int i = L - 2; i >= 0; i--) {
        auto wT = Ttranspose(*net->weights[i + 1]);
        auto err = Tmatmul(*wT, *grad[i + 1]);

        auto actPrime_i = TSigmoidPrime(*Tcopy(*zvals[i]));
        grad[i] = Tmul(*err, *actPrime_i);

        auto aT = Ttranspose(*activations[i]);
        deltaW[i] = TmulScalar(*Tmatmul(*grad[i], *aT), net->learningRate);
        deltaB[i] = TmulScalar(*grad[i], net->learningRate);
    }

    // ----------------------------
    // UPDATE PARAMETERS
    // ----------------------------
    for (int i = 0; i < L; i++) {
        net->weights[i] = Tadd(*net->weights[i], *deltaW[i]);
        net->biases[i] = Tadd(*net->biases[i], *deltaB[i]);
    }
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
