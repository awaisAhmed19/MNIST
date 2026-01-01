#include <stdio.h>
#include <stdlib.h>

#include "raylib.h"

#define INPUT_TOTAL 784
#define H1_TOTAL 1000
#define H2_TOTAL 1000
#define OUT_TOTAL 10
#define IN_VISIBLE 50
#define HID_VISIBLE 75
#define OUT_VISIBLE 10

typedef struct {
    int width;
    int height;
} AppConfig;

AppConfig app;

typedef struct {
    Vector2 pos;
    float radius;
} Node;

Node inodes[IN_VISIBLE];
Node hnodes[HID_VISIBLE];
Node h2nodes[HID_VISIBLE];
Node onodes[OUT_VISIBLE];

void InitNodes() {
    float spacing = 4.0f;
    float bottomMargin = app.height * 0.5f;

    // radius based on visible nodes
    // float topMargin = app.height * (1/(radius*IN_VISIBLE));
    // float usableHeight = app.height - topMargin - bottomMargin;
    float radius = (app.height / HID_VISIBLE) * 0.3f;
    float height = (2.0f * radius + spacing);
    float column1 = (app.height - (IN_VISIBLE * height)) / 2.0f;
    float column2 = (app.height - (HID_VISIBLE * height)) / 2.0f;
    float column3 = (app.height - (OUT_VISIBLE * height)) / 2.0f;
    // float topMargin = (app.height - column) / 2.0f;

    // X positions
    float xInput = app.width * 0.25f;
    float xHidden = app.width * 0.50f;
    float xHidden2 = app.width * 0.65f;
    float xOutput = app.width * 0.85f;

    // STRIDE = how many real neurons we skip per visible node
    int inStride = INPUT_TOTAL / IN_VISIBLE;
    int hidStride = H1_TOTAL / HID_VISIBLE;
    int outStride = OUT_TOTAL / OUT_VISIBLE;

    for (int i = 0; i < IN_VISIBLE; i++) {
        int originalIndex = i * inStride;

        inodes[i].radius = radius;
        inodes[i].pos.x = xInput;
        inodes[i].pos.y = column1 + i * (2 * radius + spacing);
    }

    for (int i = 0; i < HID_VISIBLE; i++) {
        int originalIndex = i * hidStride;

        hnodes[i].radius = radius;
        hnodes[i].pos.x = xHidden;
        hnodes[i].pos.y = column2 + i * (2 * radius + spacing);

        h2nodes[i].radius = radius;
        h2nodes[i].pos.x = xHidden2;
        h2nodes[i].pos.y = column2 + i * (2 * radius + spacing);
    }
    for (int i = 0; i < OUT_VISIBLE; i++) {
        int originalIndex = i * outStride;

        onodes[i].radius = radius;
        onodes[i].pos.x = xOutput;
        onodes[i].pos.y = column3 + i * (2 * radius + spacing);
    }
    printf("[InitNodes] radius=%.2f stride(in)=%d stride(hidden)=%d\n", radius, inStride,
           hidStride);
}

void DrawNodes() {
    int inStride = INPUT_TOTAL / IN_VISIBLE;
    int hidStride = H1_TOTAL / HID_VISIBLE;
    int outStride = OUT_TOTAL / OUT_VISIBLE;

    // Draw input nodes
    for (int i = 0; i < IN_VISIBLE; i++) DrawCircleV(inodes[i].pos, inodes[i].radius, GREEN);

    // Draw hidden nodes
    for (int j = 0; j < HID_VISIBLE; j++) DrawCircleV(hnodes[j].pos, hnodes[j].radius, RED);
    for (int j = 0; j < HID_VISIBLE; j++) DrawCircleV(h2nodes[j].pos, h2nodes[j].radius, RED);
    for (int j = 0; j < OUT_VISIBLE; j++) DrawCircleV(onodes[j].pos, onodes[j].radius, RED);

    // Draw connections
    for (int i = 0; i < IN_VISIBLE; i++) {
        for (int j = 0; j < HID_VISIBLE; j += 2) {
            // if (abs(i - j) > 5) continue;
            DrawLineEx(inodes[i].pos, hnodes[j].pos, 1.0f, DARKGRAY);
        }
    }
    for (int i = 0; i < HID_VISIBLE; i++) {
        for (int j = 0; j < HID_VISIBLE; j += 2) {
            // if (abs(i - j) > 5) continue;
            DrawLineEx(hnodes[i].pos, h2nodes[j].pos, 1.0f, DARKGRAY);
        }
    }

    for (int i = 0; i < HID_VISIBLE; i++) {
        for (int j = 0; j < OUT_VISIBLE; j += 2) {
            // if (abs(i - j) > 5) continue;
            DrawLineEx(h2nodes[i].pos, onodes[j].pos, 1.0f, DARKGRAY);
        }
    }
}

int main(void) {
    // Init minimal window to force GLFW to load monitors
    InitWindow(100, 100, "temp");

    int mon = GetCurrentMonitor();
    app.width = GetMonitorWidth(mon);
    app.height = GetMonitorHeight(mon);

    printf("Monitor=%d WIDTH=%d HEIGHT=%d\n", mon, app.width, app.height);

    // Resize instead of closing (avoids GLX crash)
    SetWindowSize(app.width, app.height);
    SetWindowTitle("Neural Network Visualizer");

    InitNodes();
    SetTargetFPS(60);

    while (!WindowShouldClose()) {
        BeginDrawing();
        ClearBackground(BLACK);
        DrawNodes();
        EndDrawing();
    }

    CloseWindow();
    return 0;
}
