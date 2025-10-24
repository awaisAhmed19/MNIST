#include "neural_network.h"

#include <filesystem>
#include <string>

NeuralNetwork* Create(int input, int hidden, int output, double lr) {
    NeuralNetwork* net;
    net->input = input;
    net->output = output;
    net->hidden = hidden;
    net->learningRate = lr;

    Matrix hidden_layer(input, output);
    Matrix output_layer(input, output);
    hidden_layer.randomize(hidden);
    output_layer.randomize(output);
    net->hiddenWeights = hidden_layer;
    net->outputWeights = output_layer;

    return net;
}

void Train(NeuralNetwork* net, Matrix& X, Matrix& Y) {
    // feed forward
    Matrix hid_in = net->hiddenWeights.dot(X);
    Matrix hid_out = hid_in.apply(Matrix::sigmoid);
    Matrix final_in = net->outputWeights.dot(hid_out);
    Matrix final_out = final_in.apply(Matrix::sigmoid);

    // err
    Matrix output_err = Y - final_out;

    Matrix output_grad = output_err * Matrix::sigmoidPrime(final_out);  // elementwise
    Matrix hid_out_T = hid_out.T();
    Matrix d_outputWeights = output_grad.dot(hid_out_T).scale(net->learningRate);
    net->outputWeights = net->outputWeights + d_outputWeights;

    Matrix outputWeights_T = net->outputWeights.T();
    Matrix hidden_err = outputWeights_T.dot(output_grad);
    Matrix hidden_grad = hidden_err * Matrix::sigmoidPrime(hid_out);  // elementwise
    Matrix X_T = X.T();
    Matrix d_hiddenWeights = hidden_grad.dot(X_T).scale(net->learningRate);
    net->hiddenWeights = net->hiddenWeights + d_hiddenWeights;
}

void Train_batch_imgs(NeuralNetwork* net, std::vector<Filer::Img>& dataset, int batch_size) {
    for (int i = 0; i < batch_size; i++) {
        if (i % 100 == 0) std::cout << "Image number: " << i;

        Filer::Img curr = dataset[i];
        Matrix Image_vec = curr.img_data.flatten(0);
        Matrix output(10, 1);
        output.m_samples[curr.label][0] = 1;
        Train(net, Image_vec, output);
    }
}

Matrix predict_img(NeuralNetwork* net, Filer::Img& img) {
    Matrix Image_vec = img.img_data.flatten(0);
    Matrix Res = predict(net, Image_vec);
    return Res;
}

double evaluate_accuracy(NeuralNetwork* net, std::vector<Filer::Img>& dataset, int n) {
    int correct = 0;

    for (int i = 0; i < n; ++i) {
        Matrix prediction = predict_img(net, dataset[i]);
        if (prediction.argmax() == dataset[i].label) {
            correct++;
        }
    }
    return 1.0 * correct / n;
}

Matrix predict(NeuralNetwork* net, Matrix& input) {
    Matrix hidden_inputs = net->hiddenWeights.dot(input);
    Matrix hidden_outputs = hidden_inputs.apply(Matrix::sigmoid);

    Matrix final_inputs = net->outputWeights.dot(hidden_outputs);
    Matrix final_outputs = final_inputs.apply(Matrix::sigmoid);

    Matrix Res = Matrix::softmax(final_outputs);
    return Res;
}

void save(const NeuralNetwork* net, const std::string& dir_name) {
    std::filesystem::path dir = dir_name;
    try {
        std::filesystem::create_directories(dir);
        std::ofstream desc(dir / "descriptor.txt");
        if (!desc.is_open()) {
            std::cerr << "Error: failed to open descriptor file.\n";
            return;
        }
        desc << net->input << "\n";
        desc << net->hidden << "\n";
        desc << net->output << "\n";
        desc << net->learningRate << "\n";
        desc.close();
        auto hidden_path = dir / "hidden.csv";
        auto output_path = dir / "output.csv";

        Filer::save_matrix(net->hiddenWeights, hidden_path.string());
        Filer::save_matrix(net->outputWeights, output_path.string());
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error saving network: " << e.what() << std::endl;
    }
}

NeuralNetwork* load(const std::string& dir_name) {
    std::filesystem::path dir = dir_name;

    if (!std::filesystem::exists(dir) || !std::filesystem::is_directory(dir)) {
        std::cerr << "Directory doesn’t exist: " << dir << std::endl;
        return nullptr;
    }

    NeuralNetwork* net = nullptr;
    try {
        std::ifstream desc(dir / "descriptor.txt");
        if (!desc.is_open()) {
            std::cerr << "Error: descriptor file missing or unreadable.\n";
            return nullptr;
        }

        int input, hidden, output;
        double lr;
        desc >> input >> hidden >> output >> lr;
        desc.close();
        auto hidden_path = dir / "hidden.csv";
        auto output_path = dir / "output.csv";

        net->hiddenWeights = Filer::load_matrix(hidden_path.string());
        net->outputWeights = Filer::load_matrix(output_path.string());
        std::cout << " Successfully loaded network from '" << dir_name << "'\n";
    } catch (const std::exception& e) {
        std::cerr << "Error loading network: " << e.what() << std::endl;
        if (net) delete net;
        return nullptr;
    }

    return net;
}
void print(const NeuralNetwork* net) {
    std::cout << "# of Inputs: " << net->input << "\n";
    std::cout << "# of Hidden: " << net->hidden << "\n";
    std::cout << "# of Output: " << net->output << "\n";
    std::cout << "Learning Rate: " << net->learningRate << "\n";

    std::cout << "Hidden Weights:\n";
    net->hiddenWeights.print();
    std::cout << "Output Weights:\n";
    net->outputWeights.print();
}
