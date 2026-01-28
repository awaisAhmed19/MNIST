
// app_state.h
#pragma once
#include "Predictor.h"
#include "nn.h"
#include "raylib.h"

extern RenderTexture2D g_canvas;
extern Rectangle canvasRect;
extern float brushSize;

extern int g_lastPrediction;
extern AppMode g_mode;
