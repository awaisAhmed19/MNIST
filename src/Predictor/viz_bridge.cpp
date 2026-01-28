
#include <algorithm>
#include <cmath>

#include "viz_bridge.h"

VizState g_viz = {};

void viz_begin() { g_viz.active = false; }

void viz_capture(int layer, const float* data, int count) {
    VizLayer* dst = nullptr;

    switch (layer) {
        case 0:
            dst = &g_viz.in;
            break;
        case 1:
            dst = &g_viz.h1;
            break;
        case 2:
            dst = &g_viz.h2;
            break;
        case 3:
            dst = &g_viz.out;
            break;
    }
    if (!dst) return;

    dst->count = count;

    float maxv = 1e-6f;
    for (int i = 0; i < count; i++) maxv = std::max(maxv, std::abs(data[i]));

    for (int i = 0; i < count; i++) dst->values[i] = std::abs(data[i]) / maxv;
}

void viz_end(int predicted) {
    g_viz.predicted = predicted;
    g_viz.active = true;
}
