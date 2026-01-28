
#include <algorithm>
#include <cmath>

#include "app_state.h"
#include "nn.h"
#include "raylib.h"
#include "viz_bridge.h"
// ---------------- OWNED STATE ----------------

Node inodes[IN_VISIBLE];
Node hnodes[HID1_VISIBLE];
Node h2nodes[HID2_VISIBLE];
Node onodes[OUT_VISIBLE];

Edge edges[7000];
int edgeCount = 0;

static NetworkLayout layout;

// ---------------- INTERNAL HELPERS ----------------

static Color LayerColor(int layer) {
    switch (layer) {
        case 0:
            return Fade(SKYBLUE, 0.7f);  // input
        case 1:
            return Fade(ORANGE, 0.7f);  // hidden 1
        case 2:
            return Fade(PURPLE, 0.7f);  // hidden 2
        case 3:
            return Fade(GREEN, 0.8f);  // output
        default:
            return Fade(WHITE, 0.5f);
    }
}
static Color ActColor(float a) {
    a = std::clamp(a, 0.0f, 1.0f);

    unsigned char base = (unsigned char)(50 + 120 * (1.0f - a));
    unsigned char green = (unsigned char)(50 + 205 * a);

    return {base, green, base, 255};
}

static void DrawLayer(Node* nodes, int n, const VizLayer& v, int layerIndex) {
    Color outline = LayerColor(layerIndex);

    for (int i = 0; i < n && i < v.count; i++) {
        nodes[i].activation = v.values[i];

        // fill = activation
        DrawCircleV(nodes[i].pos, nodes[i].radius, ActColor(nodes[i].activation));

        // outline = layer identity
        DrawCircleLinesV(nodes[i].pos, nodes[i].radius + 0.8f, outline);
    }
}

static void LayoutLayers(void) {
    layout.radius = 4.0f;
    layout.spacing = 8.7f;
    layout.layerGap = 250.0f;

    layout.netWidth = layout.layerGap * 3.0f;

    layout.netHeight =
        fmaxf(fmaxf(IN_VISIBLE, HID1_VISIBLE), fmaxf(HID2_VISIBLE, OUT_VISIBLE)) * layout.spacing;

    float centerY = layout.origin.y + layout.netHeight * 0.5f;

    float x0 = layout.origin.x;
    float x1 = x0 + layout.layerGap;
    float x2 = x1 + layout.layerGap;
    float x3 = x2 + layout.layerGap;

    auto layoutColumn = [&](Node* nodes, int count, float x) {
        float h = count * layout.spacing;
        float y0 = centerY - h * 0.5f;
        for (int i = 0; i < count; i++) {
            nodes[i] = {{x, y0 + i * layout.spacing}, layout.radius, 0.0f};
        }
    };

    layoutColumn(inodes, IN_VISIBLE, x0);
    layoutColumn(hnodes, HID1_VISIBLE, x1);
    layoutColumn(h2nodes, HID2_VISIBLE, x2);
    layoutColumn(onodes, OUT_VISIBLE, x3);
}

static void BuildEdges(void) {
    edgeCount = 0;

    auto connect = [&](Node* A, int na, Node* B, int nb) {
        for (int i = 0; i < na; i++)
            for (int j = 0; j < nb && edgeCount < 7000; j++)
                edges[edgeCount++] = {A[i].pos, B[j].pos, 0.0f};
    };

    connect(inodes, IN_VISIBLE, hnodes, HID1_VISIBLE);
    connect(hnodes, HID1_VISIBLE, h2nodes, HID2_VISIBLE);
    connect(h2nodes, HID2_VISIBLE, onodes, OUT_VISIBLE);
}

// ---------------- PUBLIC API ----------------

Rectangle GetVizRect(void) {
    return {canvasRect.x + canvasRect.width, 35, GetScreenWidth() - canvasRect.width,
            GetScreenHeight() - 35.0f};
}

void InitNetwork(void) {
    Rectangle vr = GetVizRect();

    layout.origin.x = vr.x + 50;
    layout.origin.y = vr.y + 35;

    LayoutLayers();
    BuildEdges();
}

void DrawNetwork(void) {
    Rectangle vr = GetVizRect();

    // ---- viz area border ----
    DrawRectangleLinesEx(vr, 2.0f, DARKGRAY);

    // ---- edges (background) ----
    for (int i = 0; i < edgeCount; i++) {
        DrawLineEx(edges[i].a, edges[i].b, 1.0f, (Color){100, 100, 100, 40});
    }

    // ---- base nodes (always visible) ----
    auto drawBase = [](Node* n, int count) {
        for (int i = 0; i < count; i++) {
            DrawCircleV(n[i].pos, n[i].radius, DARKGRAY);
        }
    };

    drawBase(inodes, IN_VISIBLE);
    drawBase(hnodes, HID1_VISIBLE);
    drawBase(h2nodes, HID2_VISIBLE);
    drawBase(onodes, OUT_VISIBLE);

    // ---- overlay activations (only if prediction happened) ----
    if (!g_viz.active) return;

    DrawLayer(inodes, IN_VISIBLE, g_viz.in, 0);
    DrawLayer(hnodes, HID1_VISIBLE, g_viz.h1, 1);
    DrawLayer(h2nodes, HID2_VISIBLE, g_viz.h2, 2);
    DrawLayer(onodes, OUT_VISIBLE, g_viz.out, 3);
}
