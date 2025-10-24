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
   public:
    i16 m_rows = 0;
    i16 m_cols = 0;

    double** m_samples;
    Matrix(const i16 rows, const i16 cols) : m_rows(rows), m_cols(cols) {
        m_samples = new double*[m_rows];
        // Allocate each row
        for (int i = 0; i < m_rows; ++i) {
            m_samples[i] = new double[m_cols];

            for (int j = 0; j < m_cols; ++j) m_samples[i][j] = 0.0;
        }
    }
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

    static Matrix softmax(Matrix& mat) {
        double total = 0;
        int r = mat.m_rows;
        int c = mat.m_cols;
        for (int i = 0; i < r; ++i) {
            for (int j = 0; j < c; ++j) {
                total += exp(mat.m_samples[i][j]);
            }
        }
        Matrix tmp(r, c);

        for (int i = 0; i < r; ++i) {
            for (int j = 0; j < c; ++j) {
                tmp.m_samples[i][j] = exp(mat.m_samples[i][j]) / total;
            }
        }
        return tmp;
    }
    Matrix dot(const Matrix& mat);
    Matrix apply(const std::function<double(double)>& func) const;
    Matrix scale(double n);
    Matrix addScalar(double n);
    Matrix T();
    void getShape(int& rows, int& cols) const;
    void create(int rows, int cols);
    void fill(int n);
    void print() const;

    static double sigmoid(double input) { return 1.0 / (1 + exp(-1 * input)); }

    static Matrix sigmoidPrime(const Matrix& mat) {
        Matrix sig = mat.apply(Matrix::sigmoid);
        return sig.apply([&](double x) { return x * (1.0 - x); });
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
