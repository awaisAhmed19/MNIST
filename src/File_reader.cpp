#include "File_reader.h"

void FileReader ::get_data() {
    std::ifstream file(m_filename);

    if (!file.is_open()) {
        std::cerr << "File failed to open" << std::endl;
        return;
    }

    std::string line;

    std::getline(file, line);
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string item;
        std::vector<int> image;
        int label;

        std::getline(ss, item, ',');
        label = std::stoi(item);

        while (std::getline(ss, item, ',')) {
            image.push_back(std::stoi(item));
        }
        this->m_labels.push_back(label);
        this->m_images.push_back(image);
    }
    std::cout << "Training data set from MNIST Kaggle" << std::endl;
}

int main(int argc, char* argv[]) {
    std::string filename = "../data/train.csv";
    FileReader read(filename);
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
    return 0;
}
