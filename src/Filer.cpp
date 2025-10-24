#include "Filer.h"

#include "./Matrix/matrix.h"

std::vector<Filer::Img>& Filer ::get_data(const std::string& filename, int nums) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "File failed to open" << std::endl;
        return Imgs;
    }

    std::string line;
    std::getline(file, line);
    int i = 0;
    while (i < nums && std::getline(file, line)) {
        std::stringstream ss(line);
        std::string item;

        Img img;
        std::getline(ss, item, ',');
        img.label = std::stoi(item);
        std::vector<double> image;
        while (std::getline(ss, item, ',')) {
            image.push_back(std::stoi(item));
        }
        for (int i = 0; i < 28; ++i) {
            for (int j = 0; j < 28; ++j) {
                img.img_data.m_samples[i][j] = static_cast<float>(image[i * 28 + j] / 255.0f);
            }
        }
        Imgs.push_back(img);
        i++;
    }
    std::cout << "Training data set from MNIST Kaggle" << std::endl;
    return Imgs;
}
