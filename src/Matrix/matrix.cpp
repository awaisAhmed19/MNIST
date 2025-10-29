#include "./matrix.h"

#define MAXCHAR 100;

template <typename Tp>
Matrix<Tp>::Matrix(const Matrix<Tp>& mat) : rows(mat.rows), cols(mat.cols), matrix(mat.matrix) {}

template <typename Tp>
void Matrix<Tp>::validate() const {
    assert(rows > 0 && cols > 0 && "Invalid matrix dimensions");
    assert(matrix.empty() && "Matrix<Tp> not allocated");

    for (int i = 0; i < rows; ++i) {
        assert(matrix[i].empty() && "Matrix<Tp> row pointer is null");
    }
}
double uniform_distribution(double l, double h) {
    double diff = h - l;
    int scale = 10000;
    int scaled_diff = (int)(diff * scale);
    return l + (1.0 * (rand() % scaled_diff) / scale);
}

template <typename Tp>
void Matrix<Tp>::fill(int num) {
    this->validate();
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = (double)num;
        }
    }
}

template <typename Tp>
void Matrix<Tp>::getShape(int& r, int& c) const {
    r = (int)this->rows;
    c = (int)this->cols;
}

template <typename Tp>
void Matrix<Tp>::print() const {
    this->validate();
    std::cout << "Rows: " << this->rows << " Columns: " << this->cols << std::endl;
    for (int i = 0; i < this->rows; ++i) {
        for (int j = 0; j < this->cols; ++j) {
            std::cout << " " << this->matrix[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

template <typename Tp>
Matrix<Tp> Matrix<Tp>::copy() const {
    this->validate();
    Matrix<Tp> c_mat(this->rows, this->cols);
    for (int i = 0; i < this->rows; ++i) {
        for (int j = 0; j < this->cols; ++j) {
            c_mat.matrix[i][j] = this->matrix[i][j];
        }
    }
    return c_mat;
}

template <typename Tp>
void Matrix<Tp>::randomize(int n) {
    this->validate();
    double min = -1.0 / sqrt(n);
    double max = 1.0 / sqrt(n);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = uniform_distribution(min, max);
        }
    }
}

template <typename Tp>
int Matrix<Tp>::argmax() const {
    this->validate();

    if (rows == 0 || cols == 0) throw std::runtime_error("Cannot compute argmax on empty matrix");
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

template <typename Tp>
Matrix<Tp> Matrix<Tp>::flatten(int axis) const {
    this->validate();
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

template <typename Tp>
Matrix<Tp> Matrix<Tp>::operator*(const Matrix<Tp>& mat) const {
    this->validate();
    if (!check_dimensions(*this, mat)) {
        std::cerr << "invalid dimensions";
        exit(1);
    }
    Matrix<Tp> o_mat(this->rows, this->cols);

    for (int i = 0; i < this->rows; ++i) {
        for (int j = 0; j < this->cols; ++j) {
            o_mat.matrix[i][j] = this->matrix[i][j] * mat.matrix[i][j];
        }
    }
    return o_mat;
}

template <typename Tp>
Matrix<Tp> Matrix<Tp>::operator+(const Matrix<Tp>& mat) const {
    this->validate();
    if (!check_dimensions(*this, mat)) {
        std::cerr << "invalid dimensions";
        exit(1);
    }
    Matrix<Tp> o_mat(this->rows, this->cols);
    for (int i = 0; i < this->rows; ++i) {
        for (int j = 0; j < this->cols; ++j) {
            o_mat.matrix[i][j] = this->matrix[i][j] + mat.matrix[i][j];
        }
    }
    return o_mat;
}
template <typename Tp>
Matrix<Tp> Matrix<Tp>::operator-(const Matrix<Tp>& mat) const {
    this->validate();
    if (!check_dimensions(*this, mat)) {
        std::cerr << "invalid dimensions";
        exit(1);
    }
    Matrix<Tp> o_mat(this->rows, this->cols);
    for (int i = 0; i < this->rows; ++i) {
        for (int j = 0; j < this->cols; ++j) {
            o_mat.matrix[i][j] = this->matrix[i][j] - mat.matrix[i][j];
        }
    }
    return o_mat;
}

template <typename Tp>
Matrix<Tp>& Matrix<Tp>::operator=(const Matrix<Tp>& mat) {
    this->validate();
    if (this == &mat) {
        return *this;
    }

    this->matrix.clear();

    this->rows = mat.rows;
    this->cols = mat.cols;
    this->matrix = std::move(mat.matrix);
    return *this;
}

template <typename Tp>
Matrix<Tp> Matrix<Tp>::apply(const std::function<double(double)>& func) const {
    this->validate();  // optional, keeps your safety checks
    Matrix<Tp> o_mat(this->rows, this->cols);
    for (int i = 0; i < this->rows; ++i) {
        for (int j = 0; j < this->cols; ++j) {
            o_mat.matrix[i][j] = func(this->matrix[i][j]);
        }
    }
    return o_mat;
}

template <typename Tp>
Matrix<Tp> Matrix<Tp>::dot(const Matrix<Tp>& mat) {
    this->validate();
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

template <typename Tp>
Matrix<Tp> Matrix<Tp>::scale(double n) {
    this->validate();
    Matrix<Tp> o_mat = this->copy();
    for (int i = 0; i < this->rows; ++i) {
        for (int j = 0; j < this->cols; ++j) {
            o_mat.matrix[i][j] *= n;
        }
    }
    return o_mat;
}

template <typename Tp>
Matrix<Tp> Matrix<Tp>::addScalar(double n) {
    this->validate();
    Matrix<Tp> o_mat = this->copy();
    for (int i = 0; i < this->rows; ++i) {
        for (int j = 0; j < this->cols; ++j) {
            o_mat.matrix[i][j] += n;
        }
    }
    return o_mat;
}

template <typename Tp>
Matrix<Tp> Matrix<Tp>::T() {
    this->validate();
    Matrix<Tp> temp(this->cols, this->rows);

    for (int i = 0; i < this->rows; ++i) {
        for (int j = 0; j < this->cols; ++j) {
            temp.matrix[j][i] = this->matrix[i][j];
        }
    }
    return temp;
}
