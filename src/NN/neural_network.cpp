#include "neural_network.h"

#include <string>

NeuralNetwork* Create(int input, int hidden, int output, double lr) {
    NeuralNetwork* net;
    net->input = input;
    net->output = output;
    net->hidden = hidden;
    net->learningRate = lr;

    Matrix hidden_layer;
    hidden_layer.create(input, output);
    Matrix output_layer;
    output_layer.create(input, output);
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

    Matrix output_grad = output_err * final_out.apply(Matrix::sigmoidPrime);  // elementwise
    Matrix hid_out_T = Matrix(hid_out);
    hid_out_T.T();
    Matrix d_outputWeights = output_grad.dot(hid_out_T).scale(net->learningRate);
    net->outputWeights = net->outputWeights + d_outputWeights;

    // === Backprop Hidden -> Input ===
    Matrix outputWeights_T = Matrix(net->outputWeights);
    outputWeights_T.T();
    Matrix hidden_err = outputWeights_T.dot(output_grad);
    Matrix hidden_grad = hidden_err * hid_out.apply(Matrix::sigmoidPrime);  // elementwise
    Matrix X_T = Matrix(X);
    X_T.T();
    Matrix d_hiddenWeights = hidden_grad.dot(X_T).scale(net->learningRate);
    net->hiddenWeights = net->hiddenWeights + d_hiddenWeights;
    // Matrix output_grad = output_err * final_out.apply(Matrix::sigmoidPrime);
}

void Train_batch_imgs(NeuralNetwork* net, std::vector<Filer::Img>& dataset, int batch_size);
Matrix predict_img(const NeuralNetwork* net, const Filer::Img& img);
double evaluate_accuracy(const NeuralNetwork* net, const std::vector<Filer::Img>& dataset, int n);
Matrix predict(const NeuralNetwork* net, const Matrix& input);
void save(const NeuralNetwork* net, const std::string& filename);
NeuralNetwork* load(const std::string& filename);
void print(const NeuralNetwork* net);
