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

std::unique_ptr<Tensor> Filer::load_single_image(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    std::string line;
    if (!std::getline(file, line)) {
        throw std::runtime_error("File is empty: " + filename);
    }

    std::stringstream ss(line);
    std::string item;
    std::vector<float> pixels;

    while (std::getline(ss, item, ',')) {
        if (item.empty()) continue;
        pixels.push_back(std::stof(item));
    }

    if (pixels.size() == 28 * 28 + 1) {
        pixels.erase(pixels.begin());
    }

    if (pixels.size() != 28 * 28) {
        throw std::runtime_error("Invalid pixel count: expected 784, got " +
                                 std::to_string(pixels.size()));
    }

    auto tensor = std::make_unique<Tensor>(28, 28);

    for (int i = 0; i < 28 * 28; ++i) {
        tensor->h_data[i] = pixels[i] / 255.0f;  // normalize
    }

    return tensor;
}
void Filer::save_tensor(const Tensor* t, const std::string& file_name) {
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

std::unique_ptr<Tensor> Filer::load_tensor(const std::string& file_name) {
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
