#include <iomanip>

#include "tensor.h"

// Tensor Life Cycle

Tensor* Tcreate(int r, int c) {
    Tensor* t = new Tensor(r, c);
    std::fprintf(stderr, "[Tcreate] %p (%dx%d)\n", (void*)t, r, c);
    return t;
}

void Tfree(Tensor*& t) {
    if (!t) {
        std::fprintf(stderr, "[Tfree] called on nullptr\n");
        return;
    }
    std::fprintf(stderr, "[Tfree] freeing %p\n", (void*)t);
    delete t;
    t = nullptr;
}

std::unique_ptr<Tensor> Tonehot(int label) {
    auto t = std::make_unique<Tensor>(10, 1);

    for (int i = 0; i < 10; i++) t->h_data[i] = (i == label ? 1.0f : 0.0f);

    return t;
}

inline void assert_same_shape(const Tensor* a, const Tensor* b) {
    if (!a || !b) throw std::runtime_error("Null tensor in op");
    if (a->rows != b->rows || a->cols != b->cols) throw std::runtime_error("Shape mismatch");
}
void Tresize(Tensor* t, int r, int c) {}

// CPU ops (all return NEW tensors)

std::unique_ptr<Tensor> Tcopy(const Tensor& src) {
    auto t = std::make_unique<Tensor>(src.rows, src.cols);
    int size = src.rows * src.cols;

    std::memcpy(t->h_data, src.h_data, size * sizeof(float));

    return t;
}

std::unique_ptr<Tensor> TsumCols(const Tensor& t) {
    // t is (rows x cols)
    auto out = std::make_unique<Tensor>(t.rows, 1);

    for (int r = 0; r < t.rows; r++) {
        float sum = 0.0f;
        for (int c = 0; c < t.cols; c++) {
            sum += t.h_data[r * t.cols + c];
        }
        out->h_data[r] = sum;
    }

    return out;
}
std::unique_ptr<Tensor> Tadd(const Tensor& a, const Tensor& b) {
    assert_same_shape(&a, &b);

    auto t = std::make_unique<Tensor>(a.rows, a.cols);
    int size = a.rows * a.cols;

    for (int i = 0; i < size; ++i) t->h_data[i] = a.h_data[i] + b.h_data[i];

    return t;
}

std::unique_ptr<Tensor> Tsub(const Tensor& a, const Tensor& b) {
    assert_same_shape(&a, &b);

    auto t = std::make_unique<Tensor>(a.rows, a.cols);
    int size = a.rows * a.cols;

    for (int i = 0; i < size; ++i) t->h_data[i] = a.h_data[i] - b.h_data[i];

    return t;
}

std::unique_ptr<Tensor> Tmul(const Tensor& a, const Tensor& b) {
    assert_same_shape(&a, &b);

    auto t = std::make_unique<Tensor>(a.rows, a.cols);
    int size = a.rows * a.cols;

    for (int i = 0; i < size; ++i) t->h_data[i] = a.h_data[i] * b.h_data[i];

    return t;
}

float Tdot(const Tensor& a, const Tensor& b) {
    assert_same_shape(&a, &b);

    int size = a.rows * a.cols;
    float acc = 0.0f;

    for (int i = 0; i < size; ++i) acc += a.h_data[i] * b.h_data[i];

    return acc;
}

std::unique_ptr<Tensor> Tmatmul(const Tensor& A, const Tensor& B) {
    if (A.cols != B.rows) throw std::runtime_error("Matmul shape mismatch");

    int M = A.rows;
    int K = A.cols;
    int N = B.cols;

    auto C = std::make_unique<Tensor>(M, N);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int p = 0; p < K; p++) {
                float a = A.h_data[i * K + p];
                float b = B.h_data[p * N + j];
                sum += a * b;
            }
            C->h_data[i * N + j] = sum;
        }
    }

    return C;
}

std::unique_ptr<Tensor> TmulScalar(const Tensor& in, float s) {
    auto t = std::make_unique<Tensor>(in.rows, in.cols);
    int size = in.rows * in.cols;

    for (int i = 0; i < size; ++i) t->h_data[i] = in.h_data[i] * s;

    return t;
}

std::unique_ptr<Tensor> TaddScalar(const Tensor& in, float s) {
    auto t = std::make_unique<Tensor>(in.rows, in.cols);
    int size = in.rows * in.cols;

    for (int i = 0; i < size; i++) t->h_data[i] = in.h_data[i] + s;

    return t;
}
// activations

std::unique_ptr<Tensor> TSigmoid(const Tensor& src) {
    auto out = std::make_unique<Tensor>(src.rows, src.cols);
    int size = out->size();
    for (int i = 0; i < size; i++) {
        out->h_data[i] = 1.0f / (1.0f + expf(-src.h_data[i]));
    }
    return out;
}

std::unique_ptr<Tensor> TSigmoidPrime(const Tensor& src) {
    auto out = std::make_unique<Tensor>(src.rows, src.cols);
    int size = out->size();
    for (int i = 0; i < size; i++) {
        float s = 1.0f / (1.0f + expf(-src.h_data[i]));
        out->h_data[i] = s * (1.0f - s);
    }
    return out;
}

void TRelu(Tensor& t) {
    int size = t.rows * t.cols;
    for (int i = 0; i < size; i++) {
        if (t.h_data[i] < 0) t.h_data[i] = 0.0f;
    }
}

void TReluPrime(Tensor& t) {
    int size = t.rows * t.cols;
    for (int i = 0; i < size; i++) {
        t.h_data[i] = (t.h_data[i] > 0.0f) ? 1.0f : 0.0f;
    }
}

void TSoftmaxRows(Tensor& t) {
    int m = t.rows;
    int n = t.cols;

    // Special case: single-column vector (softmax over rows)
    if (n == 1) {
        float maxv = t.h_data[0];
        for (int r = 1; r < m; ++r) maxv = std::max(maxv, t.h_data[r]);

        float sum = 0.0f;
        for (int r = 0; r < m; ++r) {
            float e = expf(t.h_data[r] - maxv);
            e = std::max(e, 1e-12f);
            t.h_data[r] = e;
            sum += e;
        }

        if (sum == 0.0f) sum = 1e-12f;
        for (int r = 0; r < m; ++r) t.h_data[r] /= sum;

        return;
    }

    // General case: softmax row-wise
    for (int r = 0; r < m; ++r) {
        float maxv = t.h_data[r * n];
        for (int j = 1; j < n; ++j) maxv = std::max(maxv, t.h_data[r * n + j]);

        float sum = 0.0f;
        for (int j = 0; j < n; ++j) {
            float e = expf(t.h_data[r * n + j] - maxv);
            e = std::max(e, 1e-12f);
            t.h_data[r * n + j] = e;
            sum += e;
        }

        if (sum == 0.0f) sum = 1e-12f;
        for (int j = 0; j < n; ++j) t.h_data[r * n + j] /= sum;
    }
}

// utilities

std::unique_ptr<Tensor> Tflatten(const Tensor& t) {
    int size = t.rows * t.cols;
    auto out = std::make_unique<Tensor>(size, 1);

    for (int i = 0; i < size; i++) out->h_data[i] = t.h_data[i];

    return out;
}

float Tuni_dist_std(float l, float h) {
    static thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(l, h);
    return dist(rng);
}

std::unique_ptr<Tensor> Ttranspose(const Tensor& t) {
    auto out = std::make_unique<Tensor>(t.cols, t.rows);

    for (int r = 0; r < t.rows; r++)
        for (int c = 0; c < t.cols; c++) out->h_data[c * t.rows + r] = t.h_data[r * t.cols + c];

    return out;
}

void TSoftmaxCols(Tensor& t) {
    int rows = t.rows;
    int cols = t.cols;

    for (int c = 0; c < cols; c++) {
        float maxv = t.h_data[c];

        // find max in this column
        for (int r = 1; r < rows; r++) {
            float v = t.h_data[r * cols + c];
            if (v > maxv) maxv = v;
        }

        // compute exps
        float sum = 0;
        for (int r = 0; r < rows; r++) {
            float e = expf(t.h_data[r * cols + c] - maxv);
            t.h_data[r * cols + c] = e;
            sum += e;
        }

        // normalize
        for (int r = 0; r < rows; r++) {
            t.h_data[r * cols + c] /= sum;
        }
    }
}
void TRandomize(Tensor& t, float fan_in) {
    if (fan_in <= 0.0f) throw std::runtime_error("fan_in must be > 0");

    // He initialization for ReLU
    float bound = sqrtf(6.0f / fan_in);

    int size = t.rows * t.cols;
    for (int i = 0; i < size; i++) t.h_data[i] = Tuni_dist_std(-bound, bound);
}

int TArgmax(const Tensor& t) {
    if (t.cols != 1) throw std::runtime_error("must be column vector");

    float maxv = t.h_data[0];
    int max_idx = 0;

    for (int i = 1; i < t.rows; i++) {
        if (t.h_data[i] > maxv) {
            maxv = t.h_data[i];
            max_idx = i;
        }
    }
    return max_idx;
}

// debugging
void TValidate(Tensor* t);
bool TCheckDimension(Tensor* t);

void TPrint(const Tensor& t) {
    std::cout << "Tensor (" << t.rows << " x " << t.cols << ")\n";

    for (int r = 0; r < t.rows; r++) {
        for (int c = 0; c < t.cols; c++) {
            std::cout << std::setw(10) << t.h_data[r * t.cols + c] << " ";
        }
        std::cout << "\n";
    }

    std::cout << std::endl;
}
