#include <algorithm>
#include <memory>

#include "./Predictor.h"
#include "viz_bridge.h"

/* ===================== GLOBAL OWNERS ===================== */

Filer filer;

NeuralNetwork* g_net;
int g_lastPrediction = -1;

/* ===================== INIT ===================== */

bool init_predictor() {
    const std::string FileDir = "../../nn-models/nnv1_96";
    g_net = load(FileDir);

    if (!g_net) {
        std::cerr << "Neural_network failed to load\n";
        return false;
    }

    std::cout << "Neural_network loaded successfully\n";
    return true;
}

std::unique_ptr<Tensor> Tmatmul_R(const Tensor& a, const Tensor& b, int layer) {
    if (a.cols != b.rows) throw std::runtime_error("Matmul shape mismatch");

    auto C = std::make_unique<Tensor>(a.rows, b.cols);

    for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < b.cols; j++) {
            float sum = 0.0f;
            for (int p = 0; p < a.cols; p++) {
                sum += a.h_data[i * a.cols + p] * b.h_data[p * b.cols + j];
            }
            C->h_data[i * b.cols + j] = sum;
        }
    }

    C->dirty_device = true;

    return C;
}

std::unique_ptr<Tensor> TaddBias_R(const Tensor& mat, const Tensor& bias, int layer) {
    if (bias.cols != 1 || bias.rows != mat.rows) throw std::runtime_error("Bias shape mismatch");

    auto out = std::make_unique<Tensor>(mat.rows, mat.cols);

    for (int r = 0; r < mat.rows; r++) {
        float b = bias.h_data[r];
        for (int c = 0; c < mat.cols; c++) {
            out->h_data[r * mat.cols + c] = mat.h_data[r * mat.cols + c] + b;
        }
    }

    out->dirty_device = true;

    return out;
}

void TRelu_R(Tensor& t, int layer) {
    for (int i = 0; i < t.rows * t.cols; i++) t.h_data[i] = fmaxf(0.0f, t.h_data[i]);

    t.dirty_device = true;
}

std::unique_ptr<Tensor> TSoftmaxCols_R(const Tensor& src, int layer) {
    auto t = std::make_unique<Tensor>(src);

    int rows = t->rows;
    int cols = t->cols;

    for (int c = 0; c < cols; c++) {
        float maxv = -INFINITY;
        for (int r = 0; r < rows; r++) maxv = std::max(maxv, t->h_data[r * cols + c]);

        float sum = 0.0f;
        for (int r = 0; r < rows; r++) {
            float e = expf(t->h_data[r * cols + c] - maxv);
            t->h_data[r * cols + c] = e;
            sum += e;
        }
        for (int r = 0; r < rows; r++) t->h_data[r * cols + c] /= sum;
    }

    t->dirty_device = true;

    return t;
}

std::unique_ptr<Tensor> Tcopy_R(const Tensor& src, int layer) {
    auto dst = std::make_unique<Tensor>(src.rows, src.cols);

    memcpy(dst->h_data, src.h_data, src.rows * src.cols * sizeof(float));

    dst->dirty_device = true;

    return dst;
}

/* ===================== FORWARD PASS ===================== */

std::unique_ptr<Tensor> predict_R(NeuralNetwork* net, Tensor* input) {
    viz_begin();

    Tensor* a = input;
    int L = net->layers.size() - 1;
    std::unique_ptr<Tensor> out;

    // INPUT (flattened)
    viz_capture(0, a->h_data, std::min(a->rows * a->cols, VIZ_IN));

    for (int i = 0; i < L; i++) {
        auto z = Tmatmul_R(*net->weights[i], *a, i);
        auto zb = TaddBias_R(*z, *net->biases[i], i);

        if (i < L - 1) {
            auto act = Tcopy_R(*zb, i);
            TRelu_R(*act, i);
            out = std::move(act);

            // HIDDEN LAYERS
            viz_capture(i + 1, out->h_data,
                        std::min(out->rows * out->cols, (i == 0 ? VIZ_H1 : VIZ_H2)));
        } else {
            out = TSoftmaxCols_R(*zb, i);

            // OUTPUT
            viz_capture(3, out->h_data, VIZ_OUT);
        }

        a = out.get();
    }

    return out;
}

/* ===================== PREDICT ENTRY ===================== */

void predict_on_save(const std::string& pred_in) {
    if (!g_net) {
        std::cerr << "Neural_network not initialized\n";
        return;
    }

    auto input = filer.load_single_image(pred_in);
    auto img = Tflatten(*input);

    auto result = predict_R(g_net, img.get());

    g_lastPrediction = TArgmax(*result);
    viz_end(g_lastPrediction);
    std::cout << "Prediction: " << g_lastPrediction << std::endl;
}
