#pragma once

#include "scene.h"
#include "tonemap.h"
#include "utilities.h"

void InitDataContainer(GuiDataContainer* guiData);
// pathtraceInit allocates every device buffer for the scene's resolution;
// call it once at startup and pathtraceFree once at exit. pathtraceReset
// clears the accumulation buffer so the next iteration starts a fresh image
// (camera moved); it allocates nothing.
void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtraceReset();
// Copies the accumulation buffer into scene->state.image. Only saveImage needs
// the host copy, so it is done on demand rather than every iteration.
void pathtraceDownloadImage();
void pathtrace(uchar4 *pbo, int frame, int iteration);

// Feature toggles (default on). Safe to flip between iterations.
void setRussianRoulette(bool enabled);
void setMaterialSort(bool enabled);

// Startup options, read once by pathtraceInit: whether the intersection stage
// runs on OptiX (false = the naive per-object kernel) and whether OptiX
// validation mode is on (slow; checks every launch).
void setOptix(bool enabled);
void setOptixValidation(bool enabled);

// View transform used by the viewport kernel. saveImage applies the same
// function on the host, see tonemap.h.
void setToneMap(ToneMapMode mode, float exposure);
