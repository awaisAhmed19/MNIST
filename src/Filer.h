#pragma once
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

class Matrix;
class Filer {
    std::vector<int> m_labels;
    std::vector<std::vector<int>> m_images;
    std::string m_filename;

   public:
    Filer(const std::string& fname) : m_filename(fname) {}
    void get_data();

    static Matrix load_matrix(const std::string file_name);
    static void save_matrix(Matrix& mat, const std::string file_name);
    std::vector<std::vector<int>> get_image_vec() { return m_images; }
    std::vector<int> get_label_vec() { return m_labels; }
};
