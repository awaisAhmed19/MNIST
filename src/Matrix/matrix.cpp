#include "./matrix.h"

#define MAXCHAR 100;

void Matrix ::create(int row, int col) {
    assert(row > 0 && col > 0 && "Matrix dimensions must be positive!");
    this->m_rows = row;
    this->m_cols = col;
    this->m_samples = new double*[this->m_rows];
    for (int i = 0; i < this->m_rows; ++i) {
        this->m_samples[i] = new double[this->m_cols];  //() does a zero initalization
    }

    assert(m_samples != nullptr && "m_samples not allocated!");
    for (int i = 0; i < this->m_rows; ++i) {
        assert(m_samples[i] != nullptr && "Row allocation failed!");
    }
}

Matrix::Matrix(const Matrix& mat) {
    m_rows = mat.m_rows;
    m_cols = mat.m_cols;

    m_samples = new double*[m_rows];
    for (int i = 0; i < m_rows; ++i) {
        m_samples[i] = new double[m_cols];
        for (int j = 0; j < m_cols; ++j) m_samples[i][j] = mat.m_samples[i][j];
    }
}

double Matrix ::sigmoid(double input) { return 1.0 / (1 + exp(-1 * input)); }

Matrix Matrix ::sigmoidPrime(const Matrix& mat) {
    Matrix mat1;
    mat1.fill(1);
    Matrix sub = mat1 - mat;
    Matrix mul = mat * sub;
    return mul;
}

Matrix Matrix ::softmax(const Matrix& mat) {
    double total = 0;
    int r = 0, c = 0;
    mat.getShape(r, c);
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {
            total += exp(mat.m_samples[i][j]);
        }
    }
    Matrix temp;

    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {
            total += exp(mat.m_samples[i][j]);
        }
    }
    return mat;
}
void Matrix::validate() const {
    assert(m_rows > 0 && m_cols > 0 && "Invalid matrix dimensions");
    assert(m_samples != nullptr && "Matrix not allocated");
    for (int i = 0; i < m_rows; ++i) {
        assert(m_samples[i] != nullptr && "Matrix row pointer is null");
    }
}
double uniform_distribution(double l, double h) {
    double diff = h - l;
    int scale = 10000;
    int scaled_diff = (int)(diff * scale);
    return l + (1.0 * (rand() % scaled_diff) / scale);
}

void Matrix ::fill(int num) {
    this->validate();
    for (int i = 0; i < m_rows; ++i) {
        for (int j = 0; j < m_cols; ++j) {
            m_samples[i][j] = (double)num;
        }
    }
}

void Matrix::getShape(int& r, int& c) const {
    r = (int)this->m_rows;
    c = (int)this->m_cols;
}

void Matrix ::print() {
    this->validate();
    std::cout << "Rows: " << this->m_rows << " Columns: " << this->m_cols << std::endl;
    for (int i = 0; i < this->m_rows; ++i) {
        for (int j = 0; j < this->m_cols; ++j) {
            std::cout << " " << this->m_samples[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

Matrix Matrix ::copy() const {
    this->validate();
    Matrix c_mat;
    c_mat.create(this->m_rows, this->m_cols);
    for (int i = 0; i < this->m_rows; ++i) {
        for (int j = 0; j < this->m_cols; ++j) {
            c_mat.m_samples[i][j] = this->m_samples[i][j];
        }
    }
    return c_mat;
}

void Matrix ::randomize(int n) {
    this->validate();
    double min = -1.0 / sqrt(n);
    double max = 1.0 / sqrt(n);

    for (int i = 0; i < m_rows; ++i) {
        for (int j = 0; j < m_cols; ++j) {
            m_samples[i][j] = uniform_distribution(min, max);
        }
    }
}
int Matrix ::argmax() {
    this->validate();
    double max_score = 0;
    int max_ind = 0;

    for (int i = 0; i < m_rows; ++i) {
        if (m_samples[i][0] > max_score) {
            max_score = m_samples[i][0];
            max_ind = i;
        }
    }
    return max_ind;
}

Matrix Matrix::flatten(int axis) const {
    this->validate();
    Matrix mat;
    mat.create(axis == 0 ? m_rows * m_cols : 1, axis == 0 ? 1 : m_rows * m_cols);
    for (int i = 0; i < m_rows; ++i)
        for (int j = 0; j < m_cols; ++j) {
            int ind = i * m_cols + j;
            if (axis == 0)
                mat.m_samples[ind][0] = m_samples[i][j];
            else
                mat.m_samples[0][ind] = m_samples[i][j];
        }
    return mat;
}

Matrix Matrix ::operator*(const Matrix& mat) const {
    this->validate();
    if (!check_dimensions(*this, mat)) {
        std::cerr << "invalid dimensions";
        exit(1);
    }
    Matrix o_mat;
    o_mat.create(this->m_rows, this->m_cols);
    for (int i = 0; i < this->m_rows; ++i) {
        for (int j = 0; j < this->m_cols; ++j) {
            o_mat.m_samples[i][j] = this->m_samples[i][j] * mat.m_samples[i][j];
        }
    }
    return o_mat;
}
Matrix Matrix ::operator+(const Matrix& mat) const {
    this->validate();
    if (!check_dimensions(*this, mat)) {
        std::cerr << "invalid dimensions";
        exit(1);
    }
    Matrix o_mat;
    o_mat.create(this->m_rows, this->m_cols);
    for (int i = 0; i < this->m_rows; ++i) {
        for (int j = 0; j < this->m_cols; ++j) {
            o_mat.m_samples[i][j] = this->m_samples[i][j] + mat.m_samples[i][j];
        }
    }
    return o_mat;
}
Matrix Matrix ::operator-(const Matrix& mat) const {
    this->validate();
    if (!check_dimensions(*this, mat)) {
        std::cerr << "invalid dimensions";
        exit(1);
    }
    Matrix o_mat;
    o_mat.create(this->m_rows, this->m_cols);
    for (int i = 0; i < this->m_rows; ++i) {
        for (int j = 0; j < this->m_cols; ++j) {
            o_mat.m_samples[i][j] = this->m_samples[i][j] - mat.m_samples[i][j];
        }
    }
    return o_mat;
}
Matrix& Matrix ::operator=(const Matrix& mat) {
    this->validate();
    if (this == &mat) {
        return *this;
    }

    for (int i = 0; i < this->m_rows; ++i) delete[] this->m_samples[i];
    delete[] this->m_samples;

    this->m_rows = mat.m_rows;
    this->m_cols = mat.m_cols;

    this->m_samples = new double*[this->m_rows];

    for (int i = 0; i < this->m_rows; ++i) {
        this->m_samples[i] = new double[this->m_cols];
        for (int j = 0; j < this->m_cols; ++j) {
            this->m_samples[i][j] = mat.m_samples[i][j];
        }
    }
    return *this;
}

Matrix Matrix ::apply(const std::function<double(double)>& func, const Matrix& mat) {
    this->validate();
    Matrix o_mat = mat.copy();
    for (int i = 0; i < mat.m_rows; ++i) {
        for (int j = 0; j < mat.m_cols; ++j) {
            o_mat.m_samples[i][j] = func(mat.m_samples[i][j]);
        }
    }
    return o_mat;
}
Matrix Matrix ::dot(const Matrix& mat) {
    this->validate();
    if (!(this->m_cols == mat.m_rows)) {
        std::cerr << "Dimension mistmatch dot:" << this->m_rows << " " << this->m_cols << " "
                  << mat.m_rows << " " << mat.m_cols;
        exit(1);
    }
    Matrix o_mat;
    o_mat.create(this->m_rows, mat.m_cols);
    for (int i = 0; i < this->m_rows; ++i) {
        for (int j = 0; j < mat.m_cols; ++j) {
            o_mat.m_samples[i][j] = 0.0;
            for (int k = 0; k < mat.m_rows; ++k) {
                o_mat.m_samples[i][j] += this->m_samples[i][k] * mat.m_samples[k][j];
            }
        }
    }
    return o_mat;
}
Matrix Matrix::scale(double n) {
    this->validate();
    Matrix o_mat = this->copy();
    for (int i = 0; i < this->m_rows; ++i) {
        for (int j = 0; j < this->m_cols; ++j) {
            o_mat.m_samples[i][j] *= n;
        }
    }
    return o_mat;
}
Matrix Matrix::addScalar(double n) {
    this->validate();
    Matrix o_mat = this->copy();
    for (int i = 0; i < this->m_rows; ++i) {
        for (int j = 0; j < this->m_cols; ++j) {
            o_mat.m_samples[i][j] += n;
        }
    }
    return o_mat;
}
void Matrix::T() {
    this->validate();
    Matrix temp;
    temp.create(this->m_cols, this->m_rows);

    for (int i = 0; i < this->m_rows; ++i) {
        for (int j = 0; j < this->m_cols; ++j) {
            temp.m_samples[j][i] = this->m_samples[i][j];
        }
    }
    *this = temp;
}
