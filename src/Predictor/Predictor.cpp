#include "./Predictor.h"
Filer filer;
void print(const std::string& file) {
    auto ten = filer.load_single_image(file);

    std::cout << "tensor: " << ten->rows << "x" << ten->cols << "\n\n";

    std::cout << std::fixed << std::setprecision(1);

    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            std::cout << std::setw(4) << ten->h_data[i * 28 + j];
        }
        std::cout << "\n";
    }
}
void predict_on_save(const std::string& pred_in) {
    const std::string FileDir = "../../nn-models/nnv1_96";

    NeuralNetwork* prednet = load(FileDir);

    if (!prednet) {
        std::cerr << "Neural_network failed to load\n";
    }

    std::cout << "Neural_network loaded successfully\n";

    auto input = filer.load_single_image(pred_in);

    // print(pred_in);  // or print(*input) if we rewrite print()

    auto img = Tflatten(*input);
    auto result = predict(prednet, img.get());

    std::cout << "Prediction: " << TArgmax(*result) << std::endl;
}
