#include "./Predictor.h"
Filer filer;

VizCallback g_viz = nullptr;
std::unique_ptr<Tensor> Tmatmul_R(const Tensor& a, const Tensor& b) {
    if (a.cols != b.rows) throw std::runtime_error("Matmul shape mismatch");

    TtoHost(const_cast<Tensor&>(a));
    TtoHost(const_cast<Tensor&>(b));

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

    if (g_viz) {
        VizEvent e{VizOp::MatMul, &a, &b, C.get()};
        g_viz(e);
    }

    return C;
}

std::unique_ptr<Tensor> TaddBias_R(const Tensor& mat, const Tensor& bias) {
    if (bias.cols != 1 || bias.rows != mat.rows) throw std::runtime_error("Bias shape mismatch");

    TtoHost(const_cast<Tensor&>(mat));
    TtoHost(const_cast<Tensor&>(bias));

    auto out = std::make_unique<Tensor>(mat.rows, mat.cols);

    for (int r = 0; r < mat.rows; r++) {
        float b = bias.h_data[r];
        for (int c = 0; c < mat.cols; c++) {
            out->h_data[r * mat.cols + c] = mat.h_data[r * mat.cols + c] + b;
        }
    }

    out->dirty_device = true;

    if (g_viz) {
        g_viz({VizOp::AddBias, &mat, &bias, out.get()});
    }

    return out;
}
void TRelu_R(Tensor& t) {
    TtoHost(t);

    for (int i = 0; i < t.rows * t.cols; i++) {
        t.h_data[i] = fmaxf(0.0f, t.h_data[i]);
    }

    t.dirty_device = true;

    if (g_viz) {
        g_viz({VizOp::ReLU, &t, nullptr, &t});
    }
}
std::unique_ptr<Tensor> TSoftmaxCols_R(const Tensor& src) {
    auto t = std::make_unique<Tensor>(src);
    TtoHost(*t);

    int rows = t->rows, cols = t->cols;

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

    if (g_viz) {
        g_viz({VizOp::Softmax, &src, nullptr, t.get()});
    }

    return t;
}
std::unique_ptr<Tensor> Tcopy_R(const Tensor& src) {
    auto dst = std::make_unique<Tensor>(src.rows, src.cols);

    TtoHost(const_cast<Tensor&>(src));
    memcpy(dst->h_data, src.h_data, src.rows * src.cols * sizeof(float));

    dst->dirty_device = true;

    if (g_viz) {
        g_viz({VizOp::Copy, &src, nullptr, dst.get()});
    }

    return dst;
}
void print(const std::string& file) {
    auto ten = filer.load_single_image(file);

    std::cout << "tensor: " << ten->rows << "x" << ten->cols << "\n\n";

    std::cout << std::fixed << std::setprecision(1);

    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            std::cout << std::setw(4) << ten->h_data[i * 28 + j];
        }
        std::cout << "\n";
    }
}

void Visualize(const VizEvent& e) {
    switch (e.op) {
        case VizOp::MatMul:
            std::cout << "[Viz] MatMul " << e.A->rows << "x" << e.A->cols << " * " << e.B->rows
                      << "x" << e.B->cols << "\n";
            break;

        case VizOp::AddBias:
            std::cout << "[Viz] AddBias\n";
            break;

        case VizOp::ReLU:
            std::cout << "[Viz] ReLU\n";
            break;

        case VizOp::Softmax:
            std::cout << "[Viz] Softmax\n";
            break;

        case VizOp::Copy:
            std::cout << "[Viz] Copy\n";
            break;
    }
}

void predict_on_save(const std::string& pred_in) {
    const std::string FileDir = "../../nn-models/nnv1_96";

    NeuralNetwork* prednet = load(FileDir);
    if (!prednet) {
        std::cerr << "Neural_network failed to load\n";
        return;
    }

    std::cout << "Neural_network loaded successfully\n";

    auto input = filer.load_single_image(pred_in);
    auto img = Tflatten(*input);

    g_viz = Visualize;  // 🔥 ENABLE VISUALIZATION

    auto result = predict(prednet, img.get());

    g_viz = nullptr;  // 🔒 DISABLE (important!)

    std::cout << "Prediction: " << TArgmax(*result) << std::endl;
}

std::unique_ptr<Tensor> predict_R(NeuralNetwork* net, Tensor* input) {
    Tensor* a = input;
    int L = net->layers.size() - 1;

    std::unique_ptr<Tensor> out;

    for (int i = 0; i < L; i++) {
        auto z = Tmatmul_R(*net->weights[i], *a);
        auto zb = TaddBias_R(*z, *net->biases[i]);

        if (i < L - 1) {
            auto act = Tcopy_R(*zb);
            TRelu(*act);
            out = std::move(act);
        } else {
            out = TSoftmaxCols_R(*zb);
        }

        a = out.get();
    }

    return out;
}
