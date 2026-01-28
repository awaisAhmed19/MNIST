

#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>

#include "Predictor.h"
#include "app_state.h"
#include "raylib.h"

// ---------------- CONFIG ----------------
#define BRUSH_MIN 2.0f
#define BRUSH_MAX 20.0f

// ---------------- INTERNAL STATE ----------------
static Vector2 prev = {-1, -1};

void SaveAndPredict(void);
// ---------------- INTERNAL HELPERS ----------------
static Rectangle FindDrawingBounds(Image img) {
    int minX = img.width, minY = img.height;
    int maxX = 0, maxY = 0;

    Color* px = LoadImageColors(img);

    for (int y = 0; y < img.height; y++) {
        for (int x = 0; x < img.width; x++) {
            Color c = px[y * img.width + x];
            if (c.r + c.g + c.b < 600) {
                minX = fminf(minX, x);
                maxX = fmaxf(maxX, x);
                minY = fminf(minY, y);
                maxY = fmaxf(maxY, y);
            }
        }
    }

    UnloadImageColors(px);

    if (minX > maxX) return {0, 0, 0, 0};

    return {(float)minX, (float)minY, (float)(maxX - minX + 1), (float)(maxY - minY + 1)};
}

// ---------------- PUBLIC API ----------------

void ClearCanvas(void) {
    BeginTextureMode(g_canvas);
    ClearBackground(RAYWHITE);
    EndTextureMode();

    g_lastPrediction = -1;
}

void UpdateCanvas(void) {
    // Brush size
    brushSize += GetMouseWheelMove() * 2.0f;
    if (brushSize < BRUSH_MIN) brushSize = BRUSH_MIN;
    if (brushSize > BRUSH_MAX) brushSize = BRUSH_MAX;

    if (IsKeyPressed(KEY_C)) ClearCanvas();
    if (IsKeyPressed(KEY_S)) SaveAndPredict();

    Vector2 m = GetMousePosition();
    bool inCanvas = CheckCollisionPointRec(m, canvasRect);

    if (inCanvas && IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
        Vector2 p = {m.x - canvasRect.x, m.y - canvasRect.y};

        BeginTextureMode(g_canvas);
        if (prev.x >= 0) DrawLineEx(prev, p, brushSize * 2, BLACK);
        DrawCircleV(p, brushSize, BLACK);
        EndTextureMode();

        prev = p;
    } else {
        prev = (Vector2){-1, -1};
    }
}

void DrawCanvas(void) {
    DrawTextureRec(g_canvas.texture,
                   {0, 0, (float)g_canvas.texture.width, -(float)g_canvas.texture.height},
                   {canvasRect.x, canvasRect.y}, WHITE);

    DrawRectangleLinesEx(canvasRect, 2, GRAY);
}

void SaveAndPredict(void) {
    Image img = LoadImageFromTexture(g_canvas.texture);
    ImageFlipVertical(&img);

    Rectangle b = FindDrawingBounds(img);
    if (b.width > 0 && b.height > 0) {
        float side = fmaxf(b.width, b.height);
        float pad = side * 0.2f;
        ImageCrop(&img, {b.x - pad, b.y - pad, side + 2 * pad, side + 2 * pad});
    }

    ImageResize(&img, 28, 28);
    ImageColorGrayscale(&img);
    ImageColorInvert(&img);
    ImageColorContrast(&img, 80);

    Color* px = LoadImageColors(img);

    std::ofstream out("drawing.csv");
    out << 0;
    for (int i = 0; i < 784; i++) out << "," << (int)px[i].r;
    out << "\n";
    out.close();

    UnloadImageColors(px);
    UnloadImage(img);

    predict_on_save(std::string("drawing.csv"));
}
