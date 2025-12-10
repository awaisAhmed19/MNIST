#include "Filer.h"
std::vector<Filer::Img> Filer::get_data(const std::string& filename, int nums) {
    std::ifstream file(filename);

    std::vector<Filer::Img> Imgs;
    if (!file.is_open()) {
        std::cerr << "File failed to open " << filename << std::endl;
        return Imgs;
    }

    std::string line;
    std::getline(file, line);

    bool has_label = line.find("label") != std::string::npos;

    int count = 0;
    while (count < nums && std::getline(file, line)) {
        std::stringstream ss(line);
        std::string item;

        Filer::Img img;
        std::vector<float> pixels;

        // --- LABEL ---
        if (has_label) {
            std::getline(ss, item, ',');
            img.label = std::stoi(item);
        }

        // --- PIXELS ---
        while (std::getline(ss, item, ',')) {
            pixels.push_back(std::stof(item));
        }

        if (pixels.size() != 28 * 28) {
            std::cerr << "Warning: invalid pixel count at row " << count
                      << " count = " << pixels.size() << std::endl;
            continue;
        }

        // Allocate Tensor 28x28 using unique_ptr
        img.img_data = std::make_unique<Tensor>(28, 28);

        for (int i = 0; i < 28 * 28; ++i) {
            img.img_data->h_data[i] = pixels[i] / 255.0f;
        }

        Imgs.push_back(std::move(img));
        count++;
    }

    std::cout << "Loaded " << Imgs.size() << " MNIST rows from " << filename << std::endl;

    return Imgs;
}
