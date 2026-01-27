#include <stdio.h>

#include "raylib.h"

#define INPUT_TOTAL 784
#define H1_TOTAL 1000
#define H2_TOTAL 1000
#define OUT_TOTAL 10

#define IN_VISIBLE 50
#define HID_VISIBLE 75
#define OUT_VISIBLE 10

#define MAX_LINKS 6
#define MAX_EDGES 8000

typedef struct {
    int width;
    int height;
} AppConfig;

typedef struct {
    Vector2 pos;
    float radius;
} Node;

typedef struct {
    Vector2 a, b;
} Edge;

typedef struct {
    float radius;
    float spacing;
    float layerGap;
    float netWidth;
    float netHeight;
    Vector2 origin;  // top-left of network bounding box
} NetworkLayout;

AppConfig app;

Node inodes[IN_VISIBLE];
Node hnodes[HID_VISIBLE];
Node h2nodes[HID_VISIBLE];
Node onodes[OUT_VISIBLE];

Edge edges[MAX_EDGES];
int edgeCount = 0;

static NetworkLayout ComputeLayout(void) {
    NetworkLayout l = {0};

    int maxVisible = HID_VISIBLE;

    l.spacing = 1.0f;
    l.radius = (app.height / (float)maxVisible) * 0.35f;
    l.layerGap = app.width * 0.12f;

    float nodeH = 2.0f * l.radius + l.spacing;
    l.netHeight = maxVisible * nodeH - l.spacing;
    l.netWidth = l.layerGap * 3.0f;

    l.origin.x = (app.width - l.netWidth) * 0.5f;
    l.origin.y = (app.height - l.netHeight) * 0.5f;

    return l;
}

static void PlaceColumn(Node* nodes, int count, float x, float centerY, float r, float spacing) {
    float nodeH = 2.0f * r + spacing;
    float colHeight = count * nodeH - spacing;
    float y0 = centerY - colHeight * 0.5f;

    for (int i = 0; i < count; i++) {
        nodes[i].radius = r;
        nodes[i].pos.x = x;
        nodes[i].pos.y = y0 + i * nodeH;
    }
}

static void AddEdge(Vector2 a, Vector2 b) {
    if (edgeCount >= MAX_EDGES) return;
    edges[edgeCount++] = (Edge){a, b};
}

static void ConnectLayers(Node* A, int aCount, Node* B, int bCount) {
    for (int i = 0; i < aCount; i++) {
        int start = (i * bCount) / aCount;
        for (int k = 0; k < MAX_LINKS && start + k < bCount; k++) {
            AddEdge(A[i].pos, B[start + k].pos);
        }
    }
}

static void InitNetwork(void) {
    NetworkLayout l = ComputeLayout();

    float cx = l.origin.x;
    float cy = l.origin.y + l.netHeight * 0.5f;

    PlaceColumn(inodes, IN_VISIBLE, cx, cy, l.radius, l.spacing);
    PlaceColumn(hnodes, HID_VISIBLE, cx + l.layerGap, cy, l.radius, l.spacing);
    PlaceColumn(h2nodes, HID_VISIBLE, cx + l.layerGap * 2.0f, cy, l.radius, l.spacing);
    PlaceColumn(onodes, OUT_VISIBLE, cx + l.layerGap * 3.0f, cy, l.radius, l.spacing);

    edgeCount = 0;
    ConnectLayers(inodes, IN_VISIBLE, hnodes, HID_VISIBLE);
    ConnectLayers(hnodes, HID_VISIBLE, h2nodes, HID_VISIBLE);
    ConnectLayers(h2nodes, HID_VISIBLE, onodes, OUT_VISIBLE);

    printf("[Network] nodes=%d edges=%d radius=%.2f\n", IN_VISIBLE + HID_VISIBLE * 2 + OUT_VISIBLE,
           edgeCount, l.radius);
}

static inline void DrawNodeArray(Node* n, int count, Color c) {
    for (int i = 0; i < count; i++) DrawCircleV(n[i].pos, n[i].radius, c);
}

static void DrawNetwork(void) {
    for (int i = 0; i < edgeCount; i++) DrawLineEx(edges[i].a, edges[i].b, 1.0f, DARKGRAY);

    DrawNodeArray(inodes, IN_VISIBLE, GREEN);
    DrawNodeArray(hnodes, HID_VISIBLE, RED);
    DrawNodeArray(h2nodes, HID_VISIBLE, RED);
    DrawNodeArray(onodes, OUT_VISIBLE, ORANGE);
}

int main(void) {
    InitWindow(100, 100, "bootstrap");

    int mon = GetCurrentMonitor();
    app.width = GetMonitorWidth(mon);
    app.height = GetMonitorHeight(mon);

    SetWindowSize(app.width, app.height);
    SetWindowTitle("Neural Network Visualizer");

    InitNetwork();
    SetTargetFPS(60);

    while (!WindowShouldClose()) {
        BeginDrawing();
        ClearBackground(BLACK);
        DrawNetwork();
        EndDrawing();
    }

    CloseWindow();
    return 0;
}
