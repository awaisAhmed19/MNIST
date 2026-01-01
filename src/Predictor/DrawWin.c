#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "Predictor.h"
#include "raylib.h"

// Compatibility helpers (for older raylib)
static float Vector2Distance(Vector2 v1, Vector2 v2) {
    float dx = v2.x - v1.x;
    float dy = v2.y - v1.y;
    return sqrtf(dx * dx + dy * dy);
}

static Vector2 Vector2Lerp(Vector2 v1, Vector2 v2, float t) {
    Vector2 result;
    result.x = v1.x + t * (v2.x - v1.x);
    result.y = v1.y + t * (v2.y - v1.y);
    return result;
}

Rectangle FindDrawingBounds(Image image) {
    int minX = image.width, minY = image.height;
    int maxX = 0, maxY = 0;

    Color* pixels = LoadImageColors(image);

    for (int y = 0; y < image.height; y++) {
        for (int x = 0; x < image.width; x++) {
            Color c = pixels[y * image.width + x];
            // Consider dark pixels as part of drawing (r+g+b < 600 to avoid pure white)
            if (c.r + c.g + c.b < 600) {
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
            }
        }
    }

    UnloadImageColors(pixels);

    if (minX > maxX) return (Rectangle){0, 0, 0, 0};  // nothing drawn

    return (Rectangle){(float)minX, (float)minY, (float)(maxX - minX + 1),
                       (float)(maxY - minY + 1)};
}

int main(void) {
    const int screenWidth = 450;
    const int screenHeight = 450;

    InitWindow(screenWidth, screenHeight, "MNIST Drawer - 28x28 Export");

    RenderTexture2D target = LoadRenderTexture(screenWidth, screenHeight);

    BeginTextureMode(target);
    ClearBackground(RAYWHITE);
    EndTextureMode();

    float brushSize = 16.0f;  // good thickness for MNIST digits
    const float BRUSH_MIN = 8.0f;
    const float BRUSH_MAX = 40.0f;

    Vector2 previousPos = {-100, -100};
    bool isDrawing = false;

    SetTargetFPS(120);

    while (!WindowShouldClose()) {
        Vector2 mousePos = GetMousePosition();

        // Brush size
        brushSize += GetMouseWheelMove() * 4.0f;
        if (brushSize < BRUSH_MIN) brushSize = BRUSH_MIN;
        if (brushSize > BRUSH_MAX) brushSize = BRUSH_MAX;

        // Clear
        if (IsKeyPressed(KEY_C)) {
            BeginTextureMode(target);
            ClearBackground(RAYWHITE);
            EndTextureMode();
            previousPos = (Vector2){-100, -100};
        }

        // Drawing logic
        if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
            isDrawing = true;
            previousPos = mousePos;
        }
        if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
            isDrawing = false;
            previousPos = (Vector2){-100, -100};
        }

        if (isDrawing && mousePos.y > 50) {
            BeginTextureMode(target);

            float dist = Vector2Distance(previousPos, mousePos);
            int steps = (int)(dist / 2.0f);
            if (steps < 1) steps = 1;

            for (int i = 1; i <= steps; i++) {
                float t = (float)i / steps;
                Vector2 pos = Vector2Lerp(previousPos, mousePos, t);
                DrawCircle((int)pos.x, (int)pos.y, brushSize, BLACK);
            }
            DrawCircle((int)mousePos.x, (int)mousePos.y, brushSize, BLACK);

            EndTextureMode();
            previousPos = mousePos;
        }

        // SAVE AS MNIST-STYLE CSV + PREVIEW (FIXED - NO DOUBLE FREE)
        if (IsKeyPressed(KEY_S)) {
            Image original = LoadImageFromTexture(target.texture);
            ImageFlipVertical(&original);

            Rectangle bounds = FindDrawingBounds(original);
            if (bounds.width > 0 && bounds.height > 0) {
                float maxSide = (bounds.width > bounds.height) ? bounds.width : bounds.height;
                float padding = maxSide * 0.2f;
                float size = maxSide + 2 * padding;

                Rectangle centered = {bounds.x - (size - bounds.width) / 2.0f,
                                      bounds.y - (size - bounds.height) / 2.0f, size, size};

                if (centered.x < 0) {
                    centered.width += centered.x;
                    centered.x = 0;
                }
                if (centered.y < 0) {
                    centered.height += centered.y;
                    centered.y = 0;
                }
                if (centered.x + centered.width > original.width)
                    centered.width = original.width - centered.x;
                if (centered.y + centered.height > original.height)
                    centered.height = original.height - centered.y;

                ImageCrop(&original, centered);
            }

            ImageResize(&original, 28, 28);
            ImageColorGrayscale(&original);
            ImageColorInvert(&original);
            ImageColorContrast(&original, 80);

            ExportImage(original, "drawing_resized.png");

            // --- Save CSV ---
            {
                Color* pixels = LoadImageColors(original);
                FILE* csv = fopen("drawing.csv", "w");
                if (csv) {
                    fprintf(csv, "0");  // placeholder label
                    for (int i = 0; i < 784; i++) {
                        fprintf(csv, ",%d", pixels[i].r);
                    }
                    fprintf(csv, "\n");
                    fclose(csv);
                }
                UnloadImageColors(pixels);
            }

            // --- Save raw binary (separate load) ---
            {
                Color* pixels = LoadImageColors(original);
                FILE* raw = fopen("drawing_28x28.raw", "wb");
                if (raw) {
                    for (int i = 0; i < 784; i++) {
                        fputc(pixels[i].r, raw);
                    }
                    fclose(raw);
                }
                UnloadImageColors(pixels);
            }

            UnloadImage(original);

            TraceLog(LOG_INFO,
                     "Saved successfully: drawing.csv + drawing_resized.png + drawing_28x28.raw");
            predict_on_save("drawing.csv");
        }
        // Draw UI
        BeginDrawing();
        ClearBackground(RAYWHITE);

        DrawTextureRec(target.texture, (Rectangle){0, 0, screenWidth, -screenHeight},
                       (Vector2){0, 0}, WHITE);

        if (mousePos.y > 50) {
            DrawCircleLines((int)mousePos.x, (int)mousePos.y, brushSize, BLACK);
        }

        DrawRectangle(0, 0, screenWidth, 50, Fade(LIGHTGRAY, 0.8f));
        DrawLine(0, 50, screenWidth, 50, GRAY);

        DrawText("MNIST Drawer | LMB: Draw | Wheel: Size | C: Clear | S: Save 28x28", 10, 10, 18,
                 DARKGRAY);
        DrawText(TextFormat("Brush: %.0f px", brushSize), 10, 30, 16, DARKGRAY);

        EndDrawing();
    }

    UnloadRenderTexture(target);
    CloseWindow();

    return 0;
}
