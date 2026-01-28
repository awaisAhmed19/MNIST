#pragma once
#include "raylib.h"

/* ===================== NETWORK SIZES ===================== */

// Semantic reference (real model)
#define INPUT_TOTAL 784
#define H1_TOTAL 512
#define H2_TOTAL 256
#define OUT_TOTAL 10

// Visualization sizes (temporary)
#define IN_VISIBLE 78
#define HID1_VISIBLE 51
#define HID2_VISIBLE 26
#define OUT_VISIBLE 10

/* ===================== APP MODE ===================== */

typedef enum { MODE_DRAW, MODE_PREDICT, MODE_VISUALIZE } AppMode;

extern AppMode g_mode;

/* ===================== VISUAL TYPES ===================== */

typedef struct {
    Vector2 pos;
    float radius;
    float activation;
} Node;

typedef struct {
    Vector2 a;
    Vector2 b;
    float weight;
} Edge;

typedef struct {
    float radius;
    float spacing;
    float layerGap;
    float netWidth;
    float netHeight;
    Vector2 origin;
} NetworkLayout;

/* ===================== EXTERNAL STATE (TEMPORARY) ===================== */

extern Node inodes[IN_VISIBLE];
extern Node hnodes[HID1_VISIBLE];
extern Node h2nodes[HID2_VISIBLE];
extern Node onodes[OUT_VISIBLE];

extern Edge edges[7000];
extern int edgeCount;

/* ===================== API ===================== */

void InitNetwork(void);
Rectangle GetVizRect(void);
void DrawNetwork(void);
