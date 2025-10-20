#pragma once

#include <cassert>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
typedef uint8_t i8;
typedef uint16_t i16;
typedef uint32_t i32;
typedef uint64_t i64;

class Matrix {
    i16 m_rows = 0;
    i16 m_cols = 0;

   public:
    double** m_samples;
    Matrix() {}
    Matrix(const Matrix& mat);
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
    Matrix& operator=(const Matrix& mat);
    void set(int r, int c, double val) {
        assert(r >= 0 && r < m_rows && c >= 0 && c < m_cols && "Matrix index out of bounds");
        m_samples[r][c] = val;
    }
    Matrix dot(const Matrix& m2);
    Matrix apply(const std::function<double(double)>& func);
    Matrix scale(double n);
    Matrix addScalar(double n);
    void T();
    void getShape(int& rows, int& cols) const;
    void create(int rows, int cols);
    void fill(int n);
    void print();
    static double sigmoid(double input) { return 1.0 / (1 + exp(-1 * input)); }

    static Matrix sigmoidPrime(const Matrix& mat) {
        Matrix sig = Matrix(mat);
        sig.apply(Matrix::sigmoid);
        Matrix ones;
        ones.create(mat.m_rows, mat.m_cols);
        ones.fill(1.0);
        return sig * (ones - sig);
    }
    Matrix softmax(const Matrix& mat);
    Matrix copy() const;
    void save(std::string file_name);
    Matrix load(std::string file_name);
    void randomize(int n);
    int argmax();
    void validate() const;
    Matrix flatten(int axis) const;
};
