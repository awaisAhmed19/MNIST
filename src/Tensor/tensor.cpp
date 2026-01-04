#include <iomanip>

#include "tensor.h"

// Tensor Life Cycle

Tensor* Tcreate(int r, int c) {
    // Constructor now allocates host + (if CUDA) device memory automatically
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

// Helpers

std::unique_ptr<Tensor> Tonehot(int label) {
    auto t = std::make_unique<Tensor>(10, 1);

    for (int i = 0; i < 10; i++) t->h_data[i] = (i == label ? 1.0f : 0.0f);

    t->dirty_device = true;
    return t;
}

inline void assert_same_shape(const Tensor* a, const Tensor* b) {
    if (!a || !b) throw std::runtime_error("Null tensor in op");
    if (a->rows != b->rows || a->cols != b->cols) throw std::runtime_error("Shape mismatch");
}

void Tresize(Tensor*, int, int) {}

// Copy

std::unique_ptr<Tensor> Tcopy(const Tensor& src) {
    auto dst = std::make_unique<Tensor>(src.rows, src.cols);

    TtoHost(const_cast<Tensor&>(src));

    memcpy(dst->h_data, src.h_data, src.rows * src.cols * sizeof(float));

    dst->dirty_device = true;
    return dst;
}

// CPU Ops (new tensors)

std::unique_ptr<Tensor> TsumCols(const Tensor& t) {
    TtoHost(const_cast<Tensor&>(t));
    auto out = std::make_unique<Tensor>(t.rows, 1);

    for (int r = 0; r < t.rows; r++) {
        float sum = 0.0f;
        for (int c = 0; c < t.cols; c++) sum += t.h_data[r * t.cols + c];
        out->h_data[r] = sum;
    }

    out->dirty_device = true;
    return out;
}

std::unique_ptr<Tensor> Tadd(const Tensor& A, const Tensor& B) {
    auto C = std::make_unique<Tensor>(A.rows, A.cols);

    TtoHost(const_cast<Tensor&>(A));
    TtoHost(const_cast<Tensor&>(B));

    for (int i = 0; i < A.rows * A.cols; i++) C->h_data[i] = A.h_data[i] + B.h_data[i];

    C->dirty_device = true;
    return C;
}

// Ops with CUDA fast path

std::unique_ptr<Tensor> Tsub(const Tensor& a, const Tensor& b) {
#ifdef USE_CUDA
    return TsubGPU(const_cast<Tensor&>(a), const_cast<Tensor&>(b));
#else
    assert_same_shape(&a, &b);
    auto t = std::make_unique<Tensor>(a.rows, a.cols);

    int n = a.rows * a.cols;
    for (int i = 0; i < n; ++i) t->h_data[i] = a.h_data[i] - b.h_data[i];

    t->dirty_device = true;
    return t;
#endif
}

std::unique_ptr<Tensor> Tmul(const Tensor& a, const Tensor& b) {
#ifdef USE_CUDA
    return TmulGPU(const_cast<Tensor&>(a), const_cast<Tensor&>(b));
#else
    assert_same_shape(&a, &b);
    auto t = std::make_unique<Tensor>(a.rows, a.cols);

    int n = a.rows * a.cols;
    for (int i = 0; i < n; ++i) t->h_data[i] = a.h_data[i] * b.h_data[i];

    t->dirty_device = true;
    return t;
#endif
}

float Tdot(const Tensor& a, const Tensor& b) {
    assert_same_shape(&a, &b);

    int n = a.rows * a.cols;
    float acc = 0.0f;

    for (int i = 0; i < n; ++i) acc += a.h_data[i] * b.h_data[i];

    return acc;
}

std::unique_ptr<Tensor> Tmatmul(const Tensor& a, const Tensor& b) {
#ifdef USE_CUDA
    return TmatmulGPU(const_cast<Tensor&>(a), const_cast<Tensor&>(b));
#else
    if (a.cols != b.rows) throw std::runtime_error("Matmul shape mismatch");

    auto C = std::make_unique<Tensor>(a.rows, b.cols);

    for (int i = 0; i < a.rows; i++)
        for (int j = 0; j < b.cols; j++) {
            float sum = 0.0f;
            for (int p = 0; p < a.cols; p++)
                sum += a.h_data[i * a.cols + p] * b.h_data[p * b.cols + j];
            C->h_data[i * b.cols + j] = sum;
        }

    C->dirty_device = true;

    return C;
#endif
}

std::unique_ptr<Tensor> TmulScalar(const Tensor& in, float s) {
#ifdef USE_CUDA
    return TscaleGPU(const_cast<Tensor&>(in), s);
#else
    auto t = std::make_unique<Tensor>(in.rows, in.cols);
    int n = in.rows * in.cols;

    for (int i = 0; i < n; ++i) t->h_data[i] = in.h_data[i] * s;

    t->dirty_device = true;
    return t;
#endif
}

std::unique_ptr<Tensor> TaddScalar(const Tensor& in, float s) {
    auto t = std::make_unique<Tensor>(in.rows, in.cols);
    int n = in.rows * in.cols;

    for (int i = 0; i < n; i++) t->h_data[i] = in.h_data[i] + s;

    t->dirty_device = true;
    return t;
}

// Activations

std::unique_ptr<Tensor> TSigmoid(const Tensor& src) {
#ifdef USE_CUDA
    return TSigmoidGPU(const_cast<Tensor&>(src));
#else
    auto out = std::make_unique<Tensor>(src.rows, src.cols);
    int n = out->size();

    for (int i = 0; i < n; i++) out->h_data[i] = 1.0f / (1.0f + expf(-src.h_data[i]));

    out->dirty_device = true;
    return out;
#endif
}

std::unique_ptr<Tensor> TSigmoidPrime(const Tensor& src) {
#ifdef USE_CUDA
    return TSigmoidPrimeGPU(const_cast<Tensor&>(src));
#else
    auto out = std::make_unique<Tensor>(src.rows, src.cols);
    int n = out->size();

    for (int i = 0; i < n; i++) {
        float s = 1.0f / (1.0f + expf(-src.h_data[i]));
        out->h_data[i] = s * (1.0f - s);
    }

    out->dirty_device = true;
    return out;
#endif
}

void TRelu(Tensor& t) {
#ifdef USE_CUDA
    TReluGPU(t);
#else
    int n = t.rows * t.cols;
    for (int i = 0; i < n; i++)
        if (t.h_data[i] < 0) t.h_data[i] = 0;
#endif
}

void TReluPrime(Tensor& t) {
#ifdef USE_CUDA
    auto tmp = TReluPrimeGPU(t);
    memcpy(t.h_data, tmp->h_data, t.rows * t.cols * sizeof(float));
    t.dirty_device = true;
#else
    int n = t.rows * t.cols;
    for (int i = 0; i < n; i++) t.h_data[i] = (t.h_data[i] > 0) ? 1.0f : 0.0f;
#endif
}

// Softmax

std::unique_ptr<Tensor> TSoftmaxRows(const Tensor& src) {
#ifdef USE_CUDA
    return TSoftmaxRowsGPU(const_cast<Tensor&>(src));
#else
    auto t = std::make_unique<Tensor>(src);
    int m = t->rows, n = t->cols;

    for (int r = 0; r < m; r++) {
        float maxv = t->h_data[r * n];
        for (int j = 1; j < n; j++) maxv = std::max(maxv, t->h_data[r * n + j]);

        float sum = 0;
        for (int j = 0; j < n; j++) {
            float e = expf(t->h_data[r * n + j] - maxv);
            t->h_data[r * n + j] = e;
            sum += e;
        }
        for (int j = 0; j < n; j++) t->h_data[r * n + j] /= sum;
    }

    t->dirty_device = true;
    return t;
#endif
}

std::unique_ptr<Tensor> TSoftmaxCols(const Tensor& src) {
#ifdef USE_CUDA
    return TSoftmaxColsGPU(const_cast<Tensor&>(src));
#else
    auto t = std::make_unique<Tensor>(src);
    int rows = t->rows, cols = t->cols;

    for (int c = 0; c < cols; c++) {
        float maxv = -INFINITY;
        for (int r = 0; r < rows; r++) maxv = std::max(maxv, t->h_data[r * cols + c]);

        float sum = 0;
        for (int r = 0; r < rows; r++) {
            float e = expf(t->h_data[r * cols + c] - maxv);
            t->h_data[r * cols + c] = e;
            sum += e;
        }
        for (int r = 0; r < rows; r++) t->h_data[r * cols + c] /= sum;
    }

    t->dirty_device = true;
    return t;
#endif
}

// Utilities

std::unique_ptr<Tensor> Tflatten(const Tensor& t) {
    int n = t.rows * t.cols;
    auto out = std::make_unique<Tensor>(n, 1);

    for (int i = 0; i < n; i++) out->h_data[i] = t.h_data[i];

    out->dirty_device = true;
    return out;
}

std::unique_ptr<Tensor> Ttranspose(const Tensor& t) {
#ifdef USE_CUDA
    return TtransposeGPU(const_cast<Tensor&>(t));
#else
    auto out = std::make_unique<Tensor>(t.cols, t.rows);

    for (int r = 0; r < t.rows; r++)
        for (int c = 0; c < t.cols; c++) out->h_data[c * t.rows + r] = t.h_data[r * t.cols + c];

    out->dirty_device = true;
    return out;
#endif
}

void TRandomize(Tensor& t, float fan_in) {
    float bound = sqrtf(6.0f / fan_in);
    int n = t.rows * t.cols;

    for (int i = 0; i < n; i++) t.h_data[i] = Tuni_dist_std(-bound, bound);

    t.dirty_device = true;
}

std::unique_ptr<Tensor> TaddBias(const Tensor& mat, const Tensor& bias) {
    if (bias.cols != 1 || bias.rows != mat.rows) throw std::runtime_error("Bias shape mismatch");

    auto out = std::make_unique<Tensor>(mat.rows, mat.cols);

    TtoHost(const_cast<Tensor&>(mat));
    TtoHost(const_cast<Tensor&>(bias));

    for (int r = 0; r < mat.rows; r++) {
        float b = bias.h_data[r];
        for (int c = 0; c < mat.cols; c++)
            out->h_data[r * mat.cols + c] = mat.h_data[r * mat.cols + c] + b;
    }

    out->dirty_device = true;
    return out;
}

float Tuni_dist_std(float l, float h) {
    static thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(l, h);
    return dist(rng);
}
int TArgmax(const Tensor& t) {
    TtoHost(const_cast<Tensor&>(t));
    int idx = 0;
    float maxv = t.h_data[0];

    for (int i = 1; i < t.rows; i++)
        if (t.h_data[i] > maxv) {
            maxv = t.h_data[i];
            idx = i;
        }

    return idx;
}

// ---------------------------
// Debug print
// ---------------------------

void TPrint(const Tensor& t) {
    TtoHost(const_cast<Tensor&>(t));
    std::cout << "Tensor (" << t.rows << " x " << t.cols << ")\n";
    for (int r = 0; r < t.rows; r++) {
        for (int c = 0; c < t.cols; c++)
            std::cout << std::setw(10) << t.h_data[r * t.cols + c] << " ";
        std::cout << "\n";
    }
}
