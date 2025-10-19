#pragma once
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "./Matrix/matrix.h"
class Matrix;
class Filer {
    std::vector<int> m_labels;
    std::vector<std::vector<int>> m_images;
    std::string m_filename;

   public:
    Filer(const std::string& fname) : m_filename(fname) {}
    void get_data();

    static void save_matrix(Matrix& mat, const std::string file_name) {
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
    static Matrix load_matrix(const std::string file_name) {
        std::ifstream file(file_name);

        if (!file.is_open()) {
            std::cerr << "File failed to open" << file_name << std::endl;
            exit(1);
        }
        int rows, cols;

        file >> rows >> cols;

        Matrix mat(rows, cols);

        for (int i = 0; i < mat.m_rows; ++i) {
            for (int j = 0; j < mat.m_cols; ++j) {
                file >> mat.m_samples[i][j];
            }
        }

        file.close();
        std::cout << "Matrix loaded:" << file_name << std::endl;
        return mat;
    }
    std::vector<std::vector<int>> get_image_vec() { return m_images; }
    std::vector<int> get_label_vec() { return m_labels; }
};
