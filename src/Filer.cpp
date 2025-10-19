#include "Filer.h"

#include "./Matrix/matrix.h"

void Filer ::get_data() {
    std::ifstream file(m_filename);

    if (!file.is_open()) {
        std::cerr << "File failed to open" << std::endl;
        return;
    }

    std::string line;

    std::getline(file, line);
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string item;
        std::vector<int> image;
        int label;

        std::getline(ss, item, ',');
        label = std::stoi(item);

        while (std::getline(ss, item, ',')) {
            image.push_back(std::stoi(item));
        }
        this->m_labels.push_back(label);
        this->m_images.push_back(image);
    }
    std::cout << "Training data set from MNIST Kaggle" << std::endl;
}
