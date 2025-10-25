#pragma once
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "../src/Matrix/matrix.h"
class Matrix;

class Filer {
    std::string m_filename;

   public:
    struct Img {
        Matrix img_data;
        int label;
        Img() : img_data(28, 28), label(0) {}
    };
    std::vector<Img> Imgs;
    Filer(const std::string& fname) : m_filename(fname) {}
    Filer() {}

    std::vector<Filer::Img> get_data(const std::string& filename, int nums);
    void print();

    static void save_matrix(const Matrix& mat, const std::string file_name) {
        std::ofstream file(file_name);

        if (!file.is_open()) {
            std::cerr << "File failed to open" << std::endl;
            return;
        }

        int rows = mat.m_rows;
        int cols = mat.m_cols;
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
                if (!(file >> mat.m_samples[i][j])) {
                    std::cerr << "Error: malformed matrix file " << file_name << std::endl;
                    exit(1);
                }
            }
        }

        file.close();
        std::cout << "Matrix loaded:" << file_name << std::endl;
        return mat;
    }
};
