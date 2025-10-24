#include <iostream>
#include <vector>

#include "./Matrix/matrix.h"
#include "./NN/neural_network.h"
#include "Filer.h"

int main(int argc, char* argv[]) {
    std::string filename = "../data/train.csv";

    // Training
    int num = 5000;
    Filer Tfile;
    std::vector<Filer::Img> Train_dataset = Tfile.get_data("../data/train.csv", num);
    NeuralNetwork* net = new NeuralNetwork(784, 300, 10, 0.1);
    Train_batch_imgs(net, Train_dataset, num);
    save(net, "testing");

    // predict
    int nums = 3000;

    Filer tfile;
    std::vector<Filer::Img> Test_dataset = tfile.get_data("../data/test.csv", nums);
    NeuralNetwork* tnet = load("testing");
    double score = evaluate_accuracy(tnet, Test_dataset, 1000);
    std::cout << "score: " << score;

    return 0;
}
