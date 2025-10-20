#include <iostream>

#include "./Matrix/matrix.h"
#include "Filer.h"
int main(int argc, char* argv[]) {
    std::string filename = "../data/train.csv";
    Filer read(filename);
    std::vector<Filer::Img> dataset;
    dataset = read.get_data();

    for (int i = 0; i < 10; i++) {
        std::cout << "label: " << dataset[i].label << std::endl;
        dataset[i].img_data.print();
    }
    /*
      Matrix mat;
      mat.create(3, 3);
      std::vector<std::vector<double>> buff = {
          {0.30, 0.03, 0.9}, {0.30, 0.03, 0.9}, {0.30, 0.03, 0.9}};
      for (int i = 0; i < 3; i++) {
          for (int j = 0; j < 3; j++) {
              mat.set(i, j, buff[i][j]);
          }
      }

      for (int i = 0; i < 3; i++) {
          for (int j = 0; j < 3; j++) {
              std::cout << mat.m_samples[i][j] << " ";
          }
          std::cout << std::endl;
      }
  */
    return 0;
}
