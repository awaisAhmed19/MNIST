#pragma once
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "../src/Matrix/matrix.h"
template <typename Tp>
class Matrix;

class Filer {
    std::string m_filename;

   public:
    struct Img {
        Matrix<float> img_data;
        int label;
        Img() : img_data(28, 28), label(0) {}
    };
    std::vector<Img> Imgs;
    Filer(const std::string& fname) : m_filename(fname) {}
    Filer() {}

    std::vector<Filer::Img> get_data(const std::string& filename, int nums);
    void print();

    static void save_matrix(const Matrix<float>& mat, const std::string file_name) {
        std::ofstream file(file_name);

        if (!file.is_open()) {
            std::cerr << "File failed to open" << std::endl;
            return;
        }

        file << mat.row() << " " << mat.col() << std::endl;

        for (int i = 0; i < mat.row(); ++i) {
            for (int j = 0; j < mat.col(); ++j) {
                file << mat.matrix[i][j];
                if (j < mat.col() - 1) file << " ";
            }
            file << std::endl;
        }

        file.close();
        std::cout << "Matrix saved in:" << file_name << std::endl;
    }

    static Matrix<float> load_matrix(const std::string file_name) {
        std::ifstream file(file_name);

        if (!file.is_open()) {
            std::cerr << "File failed to open" << file_name << std::endl;
            exit(1);
        }
        int rows, cols;

        file >> rows >> cols;

        Matrix<float> mat(rows, cols);

        for (int i = 0; i < mat.row(); ++i) {
            for (int j = 0; j < mat.col(); ++j) {
                if (!(file >> mat.matrix[i][j])) {
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
