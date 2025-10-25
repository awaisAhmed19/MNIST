#include "Filer.h"

#include "./Matrix/matrix.h"

std::vector<Filer::Img> Filer::get_data(const std::string& filename, int nums) {
    std::ifstream file(filename);

    std::vector<Filer::Img> Imgs;
    if (!file.is_open()) {
        std::cerr << "File failed to open " << filename << std::endl;
        return Imgs;
    }

    std::string line;
    std::getline(file, line);  // skip header

    bool has_label = line.find("label") != std::string::npos;

    int i = 0;
    while (i < nums && std::getline(file, line)) {
        std::stringstream ss(line);
        std::string item;

        Filer::Img img;
        std::vector<double> image;

        if (has_label) {
            std::getline(ss, item, ',');
            img.label = std::stoi(item);
        } else {
            img.label = -1;  // unknown
        }

        while (std::getline(ss, item, ',')) {
            image.push_back(std::stoi(item));
        }

        if (image.size() != 28 * 28) {
            std::cerr << "Warning: invalid pixel count in line " << i << ": " << image.size()
                      << std::endl;
            continue;
        }

        for (int r = 0; r < 28; ++r) {
            for (int c = 0; c < 28; ++c) {
                img.img_data.m_samples[r][c] = static_cast<float>(image[r * 28 + c] / 255.0f);
            }
        }

        Imgs.push_back(img);
        i++;
    }

    std::cout << "Data set from MNIST Kaggle loaded in " << filename << std::endl;
    return Imgs;
}
