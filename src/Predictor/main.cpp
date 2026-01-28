
#include "Predictor.h"
#include "app_state.h"
#include "canvas.h"
#include "nn.h"
#include "raylib.h"
#include "ui.h"
RenderTexture2D g_canvas = {};
Rectangle canvasRect = {};
float brushSize = 6.0f;
AppMode g_mode = MODE_DRAW;
void InitApp() {
    const int W = 1200;
    const int H = 800;

    InitWindow(W, H, "Draw + Predict");
    SetTargetFPS(60);

    // drawing canvas (top-left)
    float canvasW = W * 0.2f;
    float canvasH = H * 0.25f;
    canvasRect = {0, 35, canvasW, canvasH};

    g_canvas = LoadRenderTexture((int)canvasW, (int)canvasH);
    BeginTextureMode(g_canvas);
    ClearBackground(RAYWHITE);
    EndTextureMode();

    InitNetwork();
}

void UpdateApp() {
    UpdateCanvas();  // canvas.cpp
}

void DrawApp() {
    BeginDrawing();
    ClearBackground(BLACK);

    DrawTopBar();   // ui.cpp
    DrawCanvas();   // canvas.cpp
    DrawNetwork();  // network_viz.cpp

    EndDrawing();
}

void ShutdownApp() {
    UnloadRenderTexture(g_canvas);
    CloseWindow();
}

int main() {
    InitApp();
    init_predictor();  // ONCE

    while (!WindowShouldClose()) {
        UpdateApp();
        DrawApp();
    }

    ShutdownApp();
    return 0;
}
