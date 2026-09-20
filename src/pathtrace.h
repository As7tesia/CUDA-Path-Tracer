#pragma once

#include "scene.h"
#include "tonemap.h"
#include "utilities.h"

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration);

// Feature toggles (default on). Safe to flip between iterations.
void setRussianRoulette(bool enabled);

// View transform used by the viewport kernel. saveImage applies the same
// function on the host, see tonemap.h.
void setToneMap(ToneMapMode mode, float exposure);
