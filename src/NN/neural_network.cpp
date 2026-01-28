#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

#include "neural_network.h"
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

NeuralNetwork* Create(int input, int hidden, int output, float lr) {
    std::vector<int> layers = {input, hidden, output};
    return new NeuralNetwork(layers, lr);
}
std::unique_ptr<Tensor> stack_batch_inputs(const std::vector<Filer::Img>& dataset, int start,
                                           int batch_size) {
    int cols = batch_size;
    int rows = dataset[0].img_data->rows * dataset[0].img_data->cols;

    auto X = std::make_unique<Tensor>(rows, cols);

    for (int b = 0; b < batch_size; b++) {
        const Tensor& img = *dataset[start + b].img_data;

        TtoHost(const_cast<Tensor&>(img));

        for (int i = 0; i < rows; i++) X->h_data[i * cols + b] = img.h_data[i];
    }

    X->dirty_device = true;
    return X;
}

// Efficient batch label stacking: create one-hot labels directly.
std::unique_ptr<Tensor> stack_batch_labels(const std::vector<Filer::Img>& dataset, int start,
                                           int batch_size) {
    int cols = batch_size;
    auto Y = std::make_unique<Tensor>(10, cols);

    for (int b = 0; b < batch_size; b++) {
        int label = dataset[start + b].label;
        for (int i = 0; i < 10; i++) Y->h_data[i * cols + b] = (i == label ? 1.0f : 0.0f);
    }

    Y->dirty_device = true;
    return Y;
}
void Train(NeuralNetwork* net, Tensor* X, Tensor* Y) {
    auto cache = forward_pass_batch(net, X);
    auto grads = backward_pass_batch(net, cache, Y);
    update_params(net, grads);
}

ForwardCache forward_pass_batch(NeuralNetwork* net, Tensor* X) {
    int L = net->layers.size() - 1;
    ForwardCache cache;

    cache.activations.push_back(std::make_unique<Tensor>(*X));
    Tensor* a = cache.activations.back().get();

    for (int i = 0; i < L; i++) {
        auto z = TaddBias(*Tmatmul(*net->weights[i], *a), *net->biases[i]);

        cache.zvals.push_back(Tcopy(*z));

        if (i == L - 1) {
            auto soft = TSoftmaxCols(*z);
            cache.activations.push_back(std::move(soft));
        } else {
            auto act = Tcopy(*z);
            TRelu(*act);
            cache.activations.push_back(std::move(act));
        }

        a = cache.activations.back().get();
    }

    return cache;
}
BackwardCache backward_pass_batch(NeuralNetwork* net, const ForwardCache& cache, Tensor* Y) {
    int L = net->layers.size() - 1;
    BackwardCache grads;

    grads.dW.resize(L);
    grads.dB.resize(L);

    std::vector<std::unique_ptr<Tensor>> dZ(L);

    int batch = Y->cols;
    float inv = 1.0f / batch;

    // Output layer delta
    dZ[L - 1] = Tsub(*cache.activations[L], *Y);

    auto aPrevT = Ttranspose(*cache.activations[L - 1]);
    grads.dW[L - 1] = TmulScalar(*Tmatmul(*dZ[L - 1], *aPrevT), inv);

    grads.dB[L - 1] = TmulScalar(*TsumCols(*dZ[L - 1]), inv);

    // Hidden layers
    for (int i = L - 2; i >= 0; i--) {
        auto wT = Ttranspose(*net->weights[i + 1]);
        auto tmp = Tmatmul(*wT, *dZ[i + 1]);

        auto prime = Tcopy(*cache.zvals[i]);
        TReluPrime(*prime);

        dZ[i] = Tmul(*tmp, *prime);

        auto aT = Ttranspose(*cache.activations[i]);

        grads.dW[i] = TmulScalar(*Tmatmul(*dZ[i], *aT), inv);
        grads.dB[i] = TmulScalar(*TsumCols(*dZ[i]), inv);
    }

    return grads;
}
void update_params(NeuralNetwork* net, const BackwardCache& grads) {
    int L = net->layers.size() - 1;

    for (int i = 0; i < L; i++) {
        auto scaledW = TmulScalar(*grads.dW[i], net->learningRate);
        auto scaledB = TmulScalar(*grads.dB[i], net->learningRate);

#ifdef USE_CUDA
        TupdateGPU(*net->weights[i], *grads.dW[i], net->learningRate);
// #else
// Tupdate(*net->weights[i], *grads.dW[i], net->learningRate);
#endif

        net->biases[i] = Tsub(*net->biases[i], *scaledB);
    }
}
void Train_batch_imgs(NeuralNetwork* net, std::vector<Filer::Img>& dataset, int batch_size) {
    static std::mt19937 rng(std::random_device{}());
    std::shuffle(dataset.begin(), dataset.end(), rng);

    int total = dataset.size();

    for (int start = 0; start < total; start += batch_size) {
        int bs = std::min(batch_size, total - start);

        auto X = stack_batch_inputs(dataset, start, bs);
        auto Y = stack_batch_labels(dataset, start, bs);

#ifdef USE_CUDA
        Train_gpu(net, X.get(), Y.get());
#else
        auto cache = forward_pass_batch(net, X.get());
        auto grads = backward_pass_batch(net, cache, Y.get());
        update_params(net, grads);
#endif
    }
}

// TODO: turn this into gpu code as well ??

// loss = - sum_i target_i * log(pred_i + eps)
// returns scalar loss for the single sample
float cross_entropy_loss(const Tensor& prediction, const Tensor& target) {
    if (prediction.rows != target.rows || prediction.cols != target.cols) {
        throw std::runtime_error("cross_entropy_loss: shape mismatch");
    }

    TtoHost(const_cast<Tensor&>(prediction));
    TtoHost(const_cast<Tensor&>(target));
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
    // predictions: (num_classes x batch)
    // targets:     (num_classes x batch)

    const float eps = 1e-7f;
    float loss = 0.0f;

    int num_classes = predictions.rows;
    int batch = predictions.cols;

    for (int b = 0; b < batch; b++) {
        float example_loss = 0.0f;

        for (int i = 0; i < num_classes; i++) {
            float y = targets.h_data[i * batch + b];

            if (y > 0.0f) {
                float p = std::max(predictions.h_data[i * batch + b], eps);
                example_loss -= std::log(p);
            }
        }

        loss += example_loss;
    }

    return loss / batch;
}

std::unique_ptr<Tensor> predict_img(NeuralNetwork* net, Filer::Img& img) {
    auto Image_vec = Tflatten(*img.img_data);
    return predict(net, Image_vec.get());
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
        auto z = Tmatmul(*net->weights[i], *a);
        auto zb = TaddBias(*z, *net->biases[i]);

        if (i < L - 1) {
            auto act = Tcopy(*zb);
            TRelu(*act);
            out = std::move(act);
        } else {
            out = TSoftmaxCols(*zb);
        }

        a = out.get();
    }

    return out;
}

void save(const NeuralNetwork* net, const std::string& dir_name) {
    namespace fs = std::filesystem;
    fs::path dir = dir_name;

    Filer filer;
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

            filer.save_tensor(net->weights[i].get(), (dir / wFile).string());
            filer.save_tensor(net->biases[i].get(), (dir / bFile).string());
        }

        std::cout << "Network saved successfully in: " << dir << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Save error: " << e.what() << "\n";
    }
}

NeuralNetwork* load(const std::string& dir_name) {
    namespace fs = std::filesystem;
    fs::path dir = dir_name;
    Filer filer;
    if (!fs::exists(dir)) {
        std::cerr << "Directory doesn’t exist.\n";
        return nullptr;
    }

    try {
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

            auto w_raw = filer.load_tensor((dir / wFile).string());
            auto b_raw = filer.load_tensor((dir / bFile).string());

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

// Helper functions
void print_vector(const std::vector<float>& v, const std::string& name) {
    std::cout << name << ": [ ";
    for (float x : v) std::cout << x << " ";
    std::cout << "]\n";
}

void print_col(const Tensor& T, int col, const std::string& name) {
    //    sync_to_host(T);

    std::cout << name << ": [ ";

    for (int r = 0; r < T.rows; r++) {
        std::cout << T.h_data[r * T.cols + col] << " ";
    }

    std::cout << "]\n";
}
