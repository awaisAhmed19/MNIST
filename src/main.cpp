#include <filesystem>
#include <iostream>
#include <memory>
#include <vector>

#include "./Matrix/matrix.h"
#include "./NN/neural_network.h"
#include "Filer.h"

namespace fs = std::filesystem;

// Config
constexpr int TRAIN_SAMPLES = 8000;
constexpr int TEST_SAMPLES = 2000;
constexpr int EVAL_SAMPLES = 2000;
constexpr int EPOCHS = 5;
constexpr double LEARNING_RATE = 0.5;
constexpr int INPUT_SIZE = 784;
constexpr int OUTPUT_SIZE = 10;
constexpr int HIDDEN_SIZE = 800;

// Ensure a file exists before trying to use it
inline void check_file_exists(const std::string& path) {
    if (!fs::exists(path)) {
        std::cerr << "Error: File not found -> " << path << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

int main(int argc, char* argv[]) {
    const std::string project_root = PROJECT_ROOT;
    const std::string train_csv = project_root + "/data/mnist60k/train_final.csv";
    const std::string val_csv = project_root + "/data/mnist60k/val_final.csv";
    const std::string model_path = project_root + "/testing";

    check_file_exists(train_csv);
    check_file_exists(val_csv);

    Filer file;

    // Load datasets
    std::cout << "Loading training data..." << std::endl;
    auto train_data = file.get_data(train_csv, TRAIN_SAMPLES);
    std::cout << "Loaded " << train_data.size() << " training samples." << std::endl;

    std::cout << "Loading validation data..." << std::endl;
    auto val_data = file.get_data(val_csv, TEST_SAMPLES);
    std::cout << "Loaded " << val_data.size() << " validation samples." << std::endl;

    auto net = std::make_unique<NeuralNetwork>(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, LEARNING_RATE);

    double best_val_score = 0.0;

    for (int epoch = 1; epoch <= EPOCHS; ++epoch) {
        std::cout << "\nEpoch " << epoch << " / " << EPOCHS << std::endl;
        Train_batch_imgs(net.get(), train_data, TRAIN_SAMPLES);

        double val_score = evaluate_accuracy(net.get(), val_data, EVAL_SAMPLES);
        std::cout << "Validation accuracy: " << val_score << std::endl;

        if (val_score > best_val_score) {
            best_val_score = val_score;
            std::cout << "New best model! Saving to " << model_path << std::endl;
            save(net.get(), model_path);
        }
    }

    std::cout << "\nTraining complete. Best validation accuracy: " << best_val_score << std::endl;

    return 0;
}
