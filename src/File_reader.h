
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
class FileReader {
    std::vector<std::vector<int>> m_images;
    std::vector<int> m_labels;
    std::string m_filename;

   public:
    FileReader(const std::string& fname) : m_filename(fname) {}
    void get_data();

    std::vector<std::vector<int>> get_image_vec() { return m_images; }
    std::vector<int> get_label_vec() { return m_labels; }
};
