#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <vector>

#include "./NN/neural_network.h"
#include "Filer.h"

constexpr int TRAIN_SAMPLES = 8000;
constexpr int TEST_SAMPLES = 2000;
constexpr int EVAL_SAMPLES = 2000;

constexpr int EPOCHS = 10;
constexpr int BATCH_SIZE = 32;
constexpr float LEARNING_RATE = 0.05;

static const std::vector<int> LAYERS = {784, 800, 300, 100, 10};

namespace fs = std::filesystem;

inline void check_file_exists(const std::string& path) {
    if (!fs::exists(path)) {
        std::cerr << "Error: File not found -> " << path << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

int main(int argc, char* argv[]) {
    const std::string project_root = PROJECT_ROOT;

    const std::string train_csv = project_root + "/data/mnist10k/train_final.csv";
    const std::string val_csv = project_root + "/data/mnist10k/val_final.csv";
    const std::string model_dir = project_root + "/testing";

    check_file_exists(train_csv);
    check_file_exists(val_csv);

    Filer file;

    std::cout << "Loading training data...\n";
    auto train_data = file.get_data(train_csv, TRAIN_SAMPLES);

    std::cout << "Loading validation data...\n";
    auto val_data = file.get_data(val_csv, TEST_SAMPLES);

    auto net = std::make_unique<NeuralNetwork>(LAYERS, LEARNING_RATE);

    // ---------------------------------------------------------------
    // Sanity check (one sample) — verifies training pipeline is valid
    // ---------------------------------------------------------------
    {
        std::cout << "\nSanity check before training\n";

        auto& sample = train_data[0];
        auto img = Tflatten(*sample.img_data);
        auto label_onehot = Tonehot(sample.label);

        auto p_before = predict(net.get(), img.get());
        float loss_before = cross_entropy_loss(*p_before, *label_onehot);

        Train(net.get(), img.get(), label_onehot.get());

        auto p_after = predict(net.get(), img.get());
        float loss_after = cross_entropy_loss(*p_after, *label_onehot);

        std::cout << "Initial prediction: " << TArgmax(*p_before) << " label: " << sample.label
                  << "\n";

        std::cout << "Loss before: " << loss_before << "\n";
        std::cout << "Loss after : " << loss_after << "\n";
    }

    // ---------------------------------------------------------------
    // Training loop (mini-batch)
    // ---------------------------------------------------------------
    float best_val = 0.0f;
    auto total_start = std::chrono::high_resolution_clock::now();

    for (int epoch = 1; epoch <= EPOCHS; epoch++) {
        auto epoch_start = std::chrono::high_resolution_clock::now();

        std::cout << "\nEpoch " << epoch << " / " << EPOCHS << "\n";

        Train_batch_imgs(net.get(), train_data, BATCH_SIZE);

        float acc = evaluate_accuracy(net.get(), val_data, EVAL_SAMPLES);

        auto epoch_end = std::chrono::high_resolution_clock::now();
        double seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(epoch_end - epoch_start)
                .count();

        std::cout << "Validation accuracy: " << acc << "\n";
        std::cout << "Epoch time: " << seconds << " seconds\n";

        if (acc > best_val) {
            best_val = acc;
            std::cout << "New best model saved\n";
            save(net.get(), model_dir);
        }
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(total_end - total_start).count();

    std::cout << "\nTraining complete\n";
    std::cout << "Best validation accuracy: " << best_val << "\n";
    std::cout << "Total training time: " << total_seconds << " seconds\n";

    return 0;
}
