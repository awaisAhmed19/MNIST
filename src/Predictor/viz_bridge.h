
#pragma once

// max nodes you actually draw per layer
#define VIZ_IN 78
#define VIZ_H1 51
#define VIZ_H2 26
#define VIZ_OUT 10

struct VizLayer {
    int count;
    float values[80];  // normalized [0..1]
};

struct VizState {
    bool active;
    int predicted;

    VizLayer in;
    VizLayer h1;
    VizLayer h2;
    VizLayer out;
};

extern VizState g_viz;

// predictor → viz
void viz_begin();
void viz_capture(int layer, const float* data, int count);
void viz_end(int predicted);
