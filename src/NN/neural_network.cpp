#include "neural_network.h"

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

NeuralNetwork* Create(const std::vector<int>& layers, float lr) {
    return new NeuralNetwork(layers, lr);
}

void Train(NeuralNetwork* net, Matrix<float>& X, Matrix<float>& Y) {
    int L = net->layers.size() - 1;

    std::vector<Matrix<float>> activation;
    std::vector<Matrix<float>> zvals;
    activation.push_back(X);

    Matrix<float> a = X;

    // Forward pass
    for (int i = 0; i < L; i++) {
        Matrix<float> z = net->weights[i].dot(a) + net->biases[i];
        Matrix<float> a_next = z.apply(Matrix<float>::sigmoid);

        zvals.push_back(z);
        activation.push_back(a_next);

        a = a_next;
    }

    std::vector<Matrix<float>> grad;
    std::vector<Matrix<float>> deltaW;
    std::vector<Matrix<float>> deltaB;

    grad.reserve(L);
    deltaW.reserve(L);
    deltaB.reserve(L);

    for (int i = 0; i < L; ++i) {
        grad.emplace_back(net->layers[i + 1], 1);
        deltaW.emplace_back(net->layers[i + 1], net->layers[i]);
        deltaB.emplace_back(net->layers[i + 1], 1);
    }

    // Output layer
    Matrix<float> error = Y - activation[L];
    grad[L - 1] = error * Matrix<float>::sigmoidPrime(activation[L]);

    deltaW[L - 1] = grad[L - 1].dot(activation[L - 1].T()).scale(net->learningRate);
    deltaB[L - 1] = grad[L - 1].scale(net->learningRate);

    // Hidden layers
    for (int i = L - 2; i >= 0; i--) {
        Matrix<float> wT = net->weights[i + 1].T();
        Matrix<float> err = wT.dot(grad[i + 1]);

        Matrix<float> g = err * Matrix<float>::sigmoidPrime(activation[i + 1]);

        grad[i] = g;

        deltaW[i] = g.dot(activation[i].T()).scale(net->learningRate);
        deltaB[i] = g.scale(net->learningRate);
    }

    // Update
    for (int i = 0; i < L; i++) {
        net->weights[i] = net->weights[i] + deltaW[i];
        net->biases[i] = net->biases[i] + deltaB[i];
    }
}

void Train_batch_imgs(NeuralNetwork* net, std::vector<Filer::Img>& dataset, int batch_size) {
    int limit = std::min(batch_size, (int)dataset.size());

    for (int i = 0; i < limit; i++) {
        Filer::Img& curr = dataset[i];
        Matrix<float> Image_vec = curr.img_data.flatten(0);

        Matrix<float> output(10, 1);
        output.matrix[curr.label][0] = 1;

        Train(net, Image_vec, output);
    }
}

Matrix<float> predict_img(NeuralNetwork* net, Filer::Img& img) {
    Matrix<float> Image_vec = img.img_data.flatten(0);
    Matrix<float> Res = predict(net, Image_vec);
    return Res;
}

float evaluate_accuracy(NeuralNetwork* net, std::vector<Filer::Img>& dataset, int n) {
    int correct = 0;

    for (int i = 0; i < n; ++i) {
        Matrix<float> prediction = predict_img(net, dataset[i]);
        if (prediction.argmax() == dataset[i].label) {
            correct++;
        }
    }
    return 1.0 * correct / n;
}

Matrix<float> predict(NeuralNetwork* net, Matrix<float>& input) {
    Matrix<float> a = input;

    int L = net->layers.size() - 1;

    for (int i = 0; i < L; i++) {
        Matrix<float> z = net->weights[i].dot(a) + net->biases[i];
        a = z.apply(Matrix<float>::sigmoid);
    }

    return Matrix<float>::softmax(a);
}

void save(const NeuralNetwork* net, const std::string& dir_name) {
    namespace fs = std::filesystem;
    fs::path dir = dir_name;
    try {
        fs::create_directories(dir);
        std::ofstream desc(dir / "descriptor.txt");
        if (!desc.is_open()) {
            std::cerr << "Error: failed to open descriptor file.\n";
            return;
        }

        desc << net->layers.size() << "\n";
        for (int size : net->layers) {
            desc << size << "\n";
        }
        desc << net->learningRate << "\n";
        desc.close();

        for (int i = 0; i < net->layers.size() - 1; ++i) {
            std::string wFile = "weights_" + std::to_string(i) + ".csv";
            std::string bFile = "biases_" + std::to_string(i) + ".csv";

            Filer::save_matrix(net->weights[i], (dir / wFile).string());
            Filer::save_matrix(net->biases[i], (dir / bFile).string());
        }

        std::cout << "Network saved successfully in: " << dir << "\n";
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error saving network: " << e.what() << std::endl;
    }
}

NeuralNetwork* load(const std::string& dir_name) {
    namespace fs = std::filesystem;
    fs::path dir = dir_name;
    if (!fs::exists(dir)) {
        std::cerr << "Directory doesnt exist ma.\n";
        return nullptr;
    }

    try {
        std::ifstream desc(dir / "descriptor.txt");
        if (!desc.is_open()) {
            std::cerr << "Error: descriptor file missing or unreadable.\n";
            return nullptr;
        }
        int L;
        desc >> L;
        std::vector<int> layers(L);
        for (int i = 0; i < L; i++) {
            desc >> layers[i];
        }
        float lr;
        desc >> lr;
        desc.close();

        NeuralNetwork* net = new NeuralNetwork(layers, lr);

        for (int i = 0; i < net->layers.size() - 1; ++i) {
            std::string wFile = "weights_" + std::to_string(i) + ".csv";
            std::string bFile = "biases_" + std::to_string(i) + ".csv";

            net->weights[i] = Filer::load_matrix((dir / wFile).string());
            net->biases[i] = Filer::load_matrix((dir / bFile).string());
        }

        std::cout << " Successfully loaded network from '" << dir_name << "'\n";
        return net;

    } catch (const std::exception& e) {
        std::cerr << "Error loading network: " << e.what() << std::endl;
        return nullptr;
    }
}

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
}
