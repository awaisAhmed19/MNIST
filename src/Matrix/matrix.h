#pragma once

#include <cassert>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>
typedef uint8_t i8;
typedef uint16_t i16;
typedef uint32_t i32;
typedef uint64_t i64;
typedef size_t s_t;

template <typename Tp>
class Matrix {
    s_t rows = 0;
    s_t cols = 0;

   public:
    std::vector<std::vector<Tp>> matrix;
    Matrix(const i16 rows, const i16 cols)
        : rows(rows), cols(cols), matrix(rows, std::vector<Tp>(cols)) {}

    Matrix(const Matrix& mat);
    inline Tp& operator()(int r, int c) { return matrix[static_cast<s_t>(r)][static_cast<s_t>(c)]; }
    friend class Filer;

    static bool check_dimensions(const Matrix& m1, const Matrix& m2) {
        if (m1.rows == m2.rows && m1.cols == m2.cols) return true;
        return false;
    }
    Matrix operator*(const Matrix& mat) const;
    Matrix operator+(const Matrix& mat) const;
    Matrix operator-(const Matrix& mat) const;
    Matrix& operator=(const Matrix& mat);
    inline s_t row() const { return (int)rows; }
    inline s_t col() const { return (int)cols; }
    void set(int r, int c, double val) {
        assert(r >= 0 && r < rows && c >= 0 && c < cols && "Matrix index out of bounds");
        matrix[r][c] = val;
    }

    static Matrix softmax(Matrix& mat) {
        double total = 0;
        int r = mat.rows;
        int c = mat.cols;
        for (int i = 0; i < r; ++i) {
            for (int j = 0; j < c; ++j) {
                total += exp(mat.matrix[i][j]);
            }
        }
        Matrix tmp(r, c);

        for (int i = 0; i < r; ++i) {
            for (int j = 0; j < c; ++j) {
                tmp.matrix[i][j] = exp(mat.matrix[i][j]) / total;
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
    int argmax() const;
    void validate() const;
    Matrix flatten(int axis) const;
};
