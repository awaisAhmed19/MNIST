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

    std::unique_ptr<Tensor> load_single_image(const std::string& filename);
    void save_tensor(const Tensor* t, const std::string& file_name);
    std::unique_ptr<Tensor> load_tensor(const std::string& file_name);

   private:
    std::string m_filename;
};
