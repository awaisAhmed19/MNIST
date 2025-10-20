#include "Filer.h"

#include "./Matrix/matrix.h"

std::vector<Filer::Img>& Filer ::get_data() {
    std::ifstream file(m_filename);

    if (!file.is_open()) {
        std::cerr << "File failed to open" << std::endl;
        return Imgs;
    }

    std::string line;
    std::getline(file, line);
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string item;

        Img img;
        std::getline(ss, item, ',');
        img.label = std::stoi(item);
        std::vector<double> image;
        while (std::getline(ss, item, ',')) {
            image.push_back(std::stoi(item));
        }
        img.img_data.create(28, 28);
        for (int i = 0; i < 28; ++i) {
            for (int j = 0; j < 28; ++j) {
                img.img_data.m_samples[i][j] = static_cast<float>(image[i * 28 + j] / 255.0f);
            }
        }
        Imgs.push_back(img);
    }
    std::cout << "Training data set from MNIST Kaggle" << std::endl;
    return Imgs;
}

void Filer::save_matrix(Matrix& mat, const std::string file_name) {
    std::ofstream file("data.txt", std::ios::app);

    if (!file.is_open()) {
        std::cerr << "File failed to open" << std::endl;
        return;
    }

    int rows, cols;
    mat.getShape(rows, cols);
    file << rows << " " << cols << std::endl;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            file << mat.m_samples[i][j];
            if (j < cols - 1) file << " ";
        }
        file << std::endl;
    }

    file.close();
    std::cout << "Matrix saved in:" << file_name << std::endl;
}
Matrix Filer::load_matrix(const std::string file_name) {
    std::ifstream file(file_name);

    if (!file.is_open()) {
        std::cerr << "File failed to open" << file_name << std::endl;
        exit(1);
    }
    int rows, cols;

    file >> rows >> cols;

    Matrix mat;
    mat.create(rows, cols);
    for (int i = 0; i < mat.m_rows; ++i) {
        for (int j = 0; j < mat.m_cols; ++j) {
            file >> mat.m_samples[i][j];
        }
    }

    file.close();
    std::cout << "Matrix loaded:" << file_name << std::endl;
    return mat;
}
