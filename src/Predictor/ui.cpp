
#include "app_state.h"
#include "canvas.h"  // ClearCanvas(), SaveAndPredict()
#include "raylib.h"

#define TOP_BAR 35
#define BTN_W 90
#define BTN_H 25
#define PAD 10

static bool Button(Rectangle r, const char* text) {
    Vector2 m = GetMousePosition();
    bool hover = CheckCollisionPointRec(m, r);

    DrawRectangleRec(r, hover ? DARKGRAY : GRAY);
    DrawRectangleLinesEx(r, 1, BLACK);

    int tw = MeasureText(text, 18);
    DrawText(text, r.x + (r.width - tw) / 2, r.y + 6, 18, WHITE);

    return hover && IsMouseButtonPressed(MOUSE_LEFT_BUTTON);
}

void DrawTopBar(void) {
    DrawRectangle(0, 0, GetScreenWidth(), TOP_BAR, DARKGRAY);
    DrawLine(0, TOP_BAR, GetScreenWidth(), TOP_BAR, BLACK);

    Rectangle btnClear = {PAD, 4, BTN_W, BTN_H};
    Rectangle btnSave = {PAD + BTN_W + 4, 4, BTN_W, BTN_H};

    if (Button(btnClear, "Clear (C)")) ClearCanvas();
    if (Button(btnSave, "Save (S)")) SaveAndPredict();

    char buf[64];
    sprintf(buf, "Brush: %.1f", brushSize);
    DrawText(buf, 260, 12, 14, WHITE);

    if (g_lastPrediction >= 0)
        sprintf(buf, "Prediction: %d", g_lastPrediction);
    else
        sprintf(buf, "Prediction: —");

    DrawText(buf, 420, 12, 14, YELLOW);
}
