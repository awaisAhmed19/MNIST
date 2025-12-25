#pragma once
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "../src/Tensor/tensor.h"

class Filer {
   public:
    struct Img {
        std::unique_ptr<Tensor> img_data;
        int label = -1;
    };

    std::vector<Img> Imgs;

    Filer(const std::string& fname) : m_filename(fname) {}
    Filer() {}

    std::vector<Filer::Img> get_data(const std::string& filename, int nums);
    void print();

    static void save_tensor(const Tensor* t, const std::string& file_name) {
        std::ofstream file(file_name);

        if (!file.is_open()) {
            std::cerr << "File failed to open: " << file_name << "\n";
            return;
        }

        file << t->rows << "," << t->cols << "\n";

        for (int r = 0; r < t->rows; ++r) {
            for (int c = 0; c < t->cols; ++c) {
                int idx = r * t->cols + c;
                file << t->h_data[idx];
                if (c < t->cols - 1) file << ",";
            }
            file << "\n";
        }

        file.close();
    }

    static std::unique_ptr<Tensor> load_tensor(const std::string& file_name) {
        std::ifstream file(file_name);

        if (!file.is_open()) {
            std::cerr << "Failed to open tensor file: " << file_name << "\n";
            return nullptr;
        }

        std::string header;
        std::getline(file, header);

        int rows, cols;
        sscanf(header.c_str(), "%d,%d", &rows, &cols);

        auto t = std::make_unique<Tensor>(rows, cols);

        for (int r = 0; r < rows; ++r) {
            std::string line;
            std::getline(file, line);

            std::stringstream ss(line);
            std::string num;

            for (int c = 0; c < cols; ++c) {
                std::getline(ss, num, ',');
                t->h_data[r * cols + c] = std::stof(num);
            }
        }

        file.close();
        return t;
    }

   private:
    std::string m_filename;
};
