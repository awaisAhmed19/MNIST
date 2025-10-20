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
    };
    std::vector<Img> Imgs;
    Filer(const std::string& fname) : m_filename(fname) {}
    std::vector<Img>& get_data();

    void print();
    static Matrix load_matrix(const std::string file_name);
    static void save_matrix(Matrix& mat, const std::string file_name);
};
