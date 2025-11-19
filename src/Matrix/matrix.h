#pragma once

#ifdef USE_CUDA

#include <cuda_runtime_api.h>
#endif // USE_CUDA

#include <cassert>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

template <typename Tp>
class Matrix {
   public:
#ifdef USE_CUDA

    // GPU memory
    void allocate_gpu();
    void copy_to_gpu();
    void copy_from_gpu();
    void free_gpu();

    // GPU accelerated ops
    Matrix<Tp> dot_gpu(const Matrix<Tp>& other) const;

    Matrix<Tp> softmax_gpu(Matrix<Tp>& mat);
    void fill_gpu(Tp value);
    Matrix<Tp> tanh_gpu() const;
    Matrix<Tp> relu_gpu() const ;
    Matrix<Tp> scaled_gpu(Tp value) const;
#endif
    friend class Filer;

    int rows = 0;
    int cols = 0;
    std::vector<std::vector<Tp>> matrix;

    Tp* d_data = nullptr;
    Matrix<Tp>(const int rows, const int cols)
        : rows(rows), cols(cols), matrix(rows, std::vector<Tp>(cols)) {}

    Matrix() : rows(0), cols(0) {}
    Matrix(const Matrix<Tp>& mat) : rows(mat.rows), cols(mat.cols), matrix(mat.matrix) {}

    inline Tp& operator()(int r, int c) { return matrix[static_cast<int>(r)][static_cast<int>(c)]; }
    inline int row() const { return (int)rows; }
    inline int col() const { return (int)cols; }


Matrix<Tp> operator+(const Matrix<Tp>& other) const {
    validate("operator+");
    if (!check_dimensions(*this, other))
        throw std::runtime_error("Dimension mismatch in operator+");

    Matrix<Tp> out(rows, cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            out.matrix[i][j] = matrix[i][j] + other.matrix[i][j];
    return out;
}

Matrix<Tp> operator-(const Matrix<Tp>& other) const {
    validate("operator-");
    if (!check_dimensions(*this, other))
        throw std::runtime_error("Dimension mismatch in operator-");

    Matrix<Tp> out(rows, cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            out.matrix[i][j] = matrix[i][j] - other.matrix[i][j];
    return out;
}

Matrix<Tp> operator*(const Matrix<Tp>& other) const {
    validate("operator*");
    if (!check_dimensions(*this, other))
        throw std::runtime_error("Dimension mismatch in operator*");

    Matrix<Tp> out(rows, cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            out.matrix[i][j] = matrix[i][j] * other.matrix[i][j];
    return out;
}

    static bool check_dimensions(const Matrix<Tp>& m1, const Matrix<Tp>& m2) {
        if (m1.rows == m2.rows && m1.cols == m2.cols) return true;
        return false;
    }

    void set(int r, int c, float val) {
        assert(r >= 0 && r < rows && c >= 0 && c < cols && "Matrix<Tp> index out of bounds");
        matrix[r][c] = val;
    }


    static Matrix<Tp> softmax(Matrix<Tp>& mat) {
        float total = 0;
        int r = mat.rows;
        int c = mat.cols;
        for (int i = 0; i < r; ++i) {
            for (int j = 0; j < c; ++j) {
                total += exp(mat.matrix[i][j]);
            }
        }
        Matrix<Tp> tmp(r, c);

        for (int i = 0; i < r; ++i) {
            for (int j = 0; j < c; ++j) {
                tmp.matrix[i][j] = exp(mat.matrix[i][j]) / total;
            }
        }
        return tmp;
    }

    void validate(std::string func_name) const {
        if (!(rows > 0 && cols > 0)) {
            std::cerr << "Invalid dimention :" << "rows: " << rows << ": cols: " << cols << "\n"
                      << "from: " << func_name << std::endl;
            assert(false && "Invalid matrix dimensions in");
        }

        if (matrix.empty()) {
            std::cerr << "empty matrix:" << "\n" << "from: " << func_name << std::endl;
            assert(false && "empty matrix");
        }

        for (int i = 0; i < rows; ++i) {
            if (matrix[i].empty()) {
                std::cerr << "empty matrix row:" << "i+1" << "\n"
                          << "from: " << func_name << std::endl;
                assert(false && "empty matrix row");
            }
        }
    }

    void fill(int num) {
        validate("fill");
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                matrix[i][j] = (float)num;
            }
        }
    }

    void getShape(int& r, int& c) const {
        r = (int)this->rows;
        c = (int)this->cols;
    }

    float uniform_distribution(float l, float h) {
        float diff = h - l;
        int scale = 10000;
        int scaled_diff = (int)(diff * scale);
        return l + (1.0 * (rand() % scaled_diff) / scale);
    }
    void print() const {
        validate("print");

        std::cout << "Rows: " << this->rows << " Columns: " << this->cols << std::endl;
        for (int i = 0; i < this->rows; ++i) {
            for (int j = 0; j < this->cols; ++j) {
                std::cout << " " << this->matrix[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }

    Matrix<Tp> copy() const {
        validate("copy");
        Matrix<Tp> c_mat(this->rows, this->cols);
        for (int i = 0; i < this->rows; ++i) {
            for (int j = 0; j < this->cols; ++j) {
                c_mat.matrix[i][j] = this->matrix[i][j];
            }
        }
        return c_mat;
    }

    void randomize(int n) {
        validate("randomize");
        float min = -1.0 / sqrt(n);
        float max = 1.0 / sqrt(n);

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                matrix[i][j] = uniform_distribution(min, max);
            }
        }
    }

    int argmax() const {
        validate("argmax");

        if (rows == 0 || cols == 0)
            throw std::runtime_error("Cannot compute argmax on empty matrix");
        if (cols != 1) throw std::runtime_error("argmax() only valid for column vectors");

        Tp max_score = matrix[0][0];
        int max_ind = 0;

        for (int i = 1; i < rows; ++i) {
            if (matrix[i][0] > max_score) {
                max_score = matrix[i][0];
                max_ind = i;
            }
        }
        return max_ind;
    }

    Matrix<Tp> flatten(int axis) const {
        validate("flatten");
        Matrix<Tp> mat(axis == 0 ? rows * cols : 1, axis == 0 ? 1 : rows * cols);

        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j) {
                int ind = i * cols + j;
                if (axis == 0)
                    mat.matrix[ind][0] = matrix[i][j];
                else
                    mat.matrix[0][ind] = matrix[i][j];
            }
        return mat;
    }

       Matrix<Tp>& operator=(const Matrix<Tp>& mat) {
        if (this == &mat) return *this;
        // Allow assigning empty matrices — they become empty targets.
        this->rows = mat.rows;
        this->cols = mat.cols;
        this->matrix = mat.matrix;
        return *this;
    }

    Matrix<Tp> apply(const std::function<float(float)>& func) const {
        validate("apply");  // optional, keeps your safety checks
        Matrix<Tp> o_mat(this->rows, this->cols);
        for (int i = 0; i < this->rows; ++i) {
            for (int j = 0; j < this->cols; ++j) {
                o_mat.matrix[i][j] = func(this->matrix[i][j]);
            }
        }
        return o_mat;
    }

    Matrix<Tp> dot(const Matrix<Tp>& mat) {
        validate("dot");
        if (!(this->cols == mat.rows)) {
            std::cerr << "Dimension mistmatch dot:" << this->rows << " " << this->cols << " "
                      << mat.rows << " " << mat.cols;
            exit(1);
        }
        Matrix<Tp> o_mat(this->rows, mat.cols);
        for (int i = 0; i < this->rows; ++i) {
            for (int j = 0; j < mat.cols; ++j) {
                o_mat.matrix[i][j] = 0.0;
                for (int k = 0; k < mat.rows; ++k) {
                    o_mat.matrix[i][j] += this->matrix[i][k] * mat.matrix[k][j];
                }
            }
        }
        return o_mat;
    }

    Matrix<Tp> scale(float n) {
    validate("scale");
        Matrix<Tp> o_mat = this->copy();
        for (int i = 0; i < this->rows; ++i) {
            for (int j = 0; j < this->cols; ++j) {
                o_mat.matrix[i][j] *= n;
            }
        }
        return o_mat;
    }

    Matrix<Tp> addScalar(float n) {
        validate("addScalar");
        Matrix<Tp> o_mat = this->copy();
        for (int i = 0; i < this->rows; ++i) {
            for (int j = 0; j < this->cols; ++j) {
                o_mat.matrix[i][j] += n;
            }
        }
        return o_mat;
    }

    Matrix<Tp> T() {
        validate("Transpose");
        Matrix<Tp> temp(this->cols, this->rows);

        for (int i = 0; i < this->rows; ++i) {
            for (int j = 0; j < this->cols; ++j) {
                temp.matrix[j][i] = this->matrix[i][j];
            }
        }
        return temp;
    }

    static float sigmoid(float input) { return 1.0 / (1 + exp(-1 * input)); }

    static float tanh(float x) {
        float nu=exp(x)-exp(-x);
        float de=exp(x)+exp(-x);
        return nu/de;
    }

    static float relu(float x) {return fmax(0,x);}

    static Matrix<Tp> sigmoidPrime(const Matrix<Tp>& mat) {
        Matrix<Tp> sig = mat.apply(Matrix<Tp>::sigmoid);
        return sig.apply([&](float x) { return x * (1.0 - x); });
    }

    void save(std::string file_name);
    Matrix<Tp> load(std::string file_name);
};
