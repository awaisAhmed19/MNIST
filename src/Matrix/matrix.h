#pragma once

#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>

#include "../Filer.h"

typedef uint8_t i8;
typedef uint16_t i16;
typedef uint32_t i32;
typedef uint64_t i64;

class Matrix {
    i16 m_rows;
    i16 m_cols;

    double** m_samples;

   public:
    Matrix(int row, int col) : m_rows(row), m_cols(col) {
        m_samples = new double*[row];
        for (int i = 0; i < row; ++i) {
            m_samples[i] = new double[col];  //() does a zero initalization
        }
    }

    ~Matrix() {
        for (int i = 0; i < m_rows; ++i) {
            delete[] m_samples[i];
        }
        delete[] m_samples;
    }

    friend class Filer;
    static bool check_dimensions(const Matrix& m1, const Matrix& m2) {
        if (m1.m_rows == m2.m_rows && m1.m_cols == m2.m_cols) return true;
        return false;
    }
    Matrix operator*(const Matrix& mat) const;
    Matrix operator+(const Matrix& mat) const;
    Matrix operator-(const Matrix& mat) const;
    Matrix operator=(const Matrix& mat);
    Matrix dot(const Matrix& m2);
    Matrix apply(const std::function<double(double)>& func, const Matrix& mat);
    Matrix scale(double n);
    Matrix addScalar(double n);
    void T();
    void getShape(int& rows, int& cols);
    Matrix create(int rows, int cols);
    void fill(int n);
    void print();

    Matrix copy() const;
    void save(std::string file_name);
    Matrix load(std::string file_name);
    void randomize(int n);
    int argmax();
    Matrix flatten(int axis) const;
};
