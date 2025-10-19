#include <cmath>

#include "./Matrix/matrix.h"
#include "Filer.h"
int main(int argc, char* argv[]) {
    /*
    std::string filename = "../data/train.csv";
    Filer read(filename);
    read.get_data();

    std::vector<std::vector<int>> images = read.get_image_vec();
    std::vector<int> labels = read.get_label_vec();
    std::cout << "loaded " << images.size() << " m_images" << std::endl;
    std::cout << "first image label " << labels[0] << std::endl;
    std::cout << "pixel values " << std::endl;

    float threshold = 0.5f;
    std::vector<int> img;
    for (int i = 0; i < (int)images[0].size(); ++i) {
        float pixel = static_cast<float>(images[420][i]) / 255.0f;
        int p = (pixel >= threshold) ? 1 : 0;
        img.push_back(p);
    }

    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            char val = (img[i * 28 + j] == 1) ? '#' : '.';
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    */

    Matrix m(23, 23);
    m.fill(5);

    Matrix result = m.apply([](double x) { return sqrt(x); }, m);
    result.print();  // prints sin(0.5) in all cells

    return 0;
}
