#include "neural_network.h"

#include <filesystem>
#include <string>

NeuralNetwork* Create(int input, int hidden, int output, double lr) {
    NeuralNetwork* net;
    net->input = input;
    net->output = output;
    net->hidden = hidden;
    net->learningRate = lr;

    Matrix<double> hidden_layer(input, output);
    Matrix<double> output_layer(input, output);
    hidden_layer.randomize(hidden);
    output_layer.randomize(output);
    net->hiddenWeights = hidden_layer;
    net->outputWeights = output_layer;

    return net;
}

void Train(NeuralNetwork* net, Matrix<double>& X, Matrix<double>& Y) {
    // feed forward
    Matrix<double> hid_in = net->hiddenWeights.dot(X);
    Matrix<double> hid_out = hid_in.apply(Matrix<double>::sigmoid);
    Matrix<double> final_in = net->outputWeights.dot(hid_out);
    Matrix<double> final_out = final_in.apply(Matrix<double>::sigmoid);

    // err
    Matrix<double> output_err = Y - final_out;

    Matrix<double> output_grad =
        output_err * Matrix<double>::sigmoidPrime(final_out);  // elementwise
    Matrix<double> hid_out_T = hid_out.T();
    Matrix<double> d_outputWeights = output_grad.dot(hid_out_T).scale(net->learningRate);
    net->outputWeights = net->outputWeights + d_outputWeights;

    Matrix<double> outputWeights_T = net->outputWeights.T();
    Matrix<double> hidden_err = outputWeights_T.dot(output_grad);
    Matrix<double> hidden_grad = hidden_err * Matrix<double>::sigmoidPrime(hid_out);  // elementwise
    Matrix<double> X_T = X.T();
    Matrix<double> d_hiddenWeights = hidden_grad.dot(X_T).scale(net->learningRate);
    net->hiddenWeights = net->hiddenWeights + d_hiddenWeights;
}

void Train_batch_imgs(NeuralNetwork* net, std::vector<Filer::Img>& dataset, int batch_size) {
    for (int i = 0; i < batch_size; i++) {
        // if (i % 100 == 0) std::cout << "Image number: " << i << std::endl;

        Filer::Img curr = dataset[i];
        Matrix<double> Image_vec = curr.img_data.flatten(0);
        Matrix<double> output(10, 1);
        output.matrix[curr.label][0] = 1;
        Train(net, Image_vec, output);
    }
}

Matrix<double> predict_img(NeuralNetwork* net, Filer::Img& img) {
    Matrix<double> Image_vec = img.img_data.flatten(0);
    Matrix<double> Res = predict(net, Image_vec);
    return Res;
}

double evaluate_accuracy(NeuralNetwork* net, std::vector<Filer::Img>& dataset, int n) {
    int correct = 0;

    for (int i = 0; i < n; ++i) {
        Matrix<double> prediction = predict_img(net, dataset[i]);
        if (prediction.argmax() == dataset[i].label) {
            correct++;
        }
    }
    return 1.0 * correct / n;
}

Matrix<double> predict(NeuralNetwork* net, Matrix<double>& input) {
    Matrix<double> hidden_inputs = net->hiddenWeights.dot(input);
    Matrix<double> hidden_outputs = hidden_inputs.apply(Matrix<double>::sigmoid);

    Matrix<double> final_inputs = net->outputWeights.dot(hidden_outputs);
    Matrix<double> final_outputs = final_inputs.apply(Matrix<double>::sigmoid);

    Matrix<double> Res = Matrix<double>::softmax(final_outputs);
    return Res;
}

void save(const NeuralNetwork* net, const std::string& dir_name) {
    std::filesystem::path dir = dir_name;
    try {
        std::filesystem::create_directories(dir);  // <-- FIXED
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

        std::cout << "Network saved successfully in: " << dir << "\n";
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

        NeuralNetwork* net = new NeuralNetwork(input, hidden, output, lr);

        auto hidden_path = dir / "hidden.csv";
        auto output_path = dir / "output.csv";

        net->hiddenWeights = Filer::load_matrix(hidden_path.string());
        net->outputWeights = Filer::load_matrix(output_path.string());

        std::cout << " Successfully loaded network from '" << dir_name << "'\n";
        return net;

    } catch (const std::exception& e) {
        std::cerr << "Error loading network: " << e.what() << std::endl;
        return nullptr;
    }
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
