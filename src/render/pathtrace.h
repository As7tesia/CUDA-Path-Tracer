#pragma once

#include "scene/scene.h"
#include "render/tonemap.h"

// What pathtrace() reports to the viewport's ImGui panel.
struct GuiDataContainer
{
    int tracedDepth = 0;
};

// Where pathtrace() writes its report; nothing is written until this is set
// (headless renders never set it).
void setGuiData(GuiDataContainer* data);
// pathtraceInit allocates every device buffer for the scene's resolution and
// pathtraceFree releases them all; the window calls the pair again when it
// switches scenes. pathtraceReset clears the accumulation buffer so the next
// iteration starts a fresh image (camera moved); it allocates nothing.
void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtraceReset();
// Replaces the environment pathtraceInit uploaded with env, for the window's
// environment picker. The caller restarts the image.
void pathtraceSetEnvironment(const Environment& env);
// Copies the accumulation buffer into scene->state.image. Only saveImage needs
// the host copy, so it is done on demand rather than every iteration.
void pathtraceDownloadImage();
// One sample per pixel, added to the accumulation buffer. With a pixel
// buffer (the viewport; null when headless) it also writes the tone-mapped
// average of iterations 1 to iteration into it.
void pathtrace(uchar4 *pbo, int iteration);

// Feature toggles (default on). Safe to flip between iterations.
void setRussianRoulette(bool enabled);
void setMaterialSort(bool enabled);
// Next event estimation with MIS: a light sample and a shadow ray at every
// hit. Off, lights are found only by BSDF sampling (the paths have to hit
// them). Takes effect only where pathtraceNeeAvailable says it can.
void setNextEventEstimation(bool enabled);
// Whether the loaded scene can use next event estimation: the shadow rays
// need OptiX, and the scene needs a light it can sample (an emissive cube or
// mesh, or a punctual light).
bool pathtraceNeeAvailable();

// Startup options, read once by pathtraceInit: whether the intersection stage
// runs on OptiX (false = the naive per-object kernel) and whether OptiX
// validation mode is on (slow; checks every launch).
void setOptix(bool enabled);
void setOptixValidation(bool enabled);

// View transform used by the viewport kernel. saveImage applies the same
// function on the host, see tonemap.h.
void setToneMap(ToneMapMode mode, float exposure);

// With --timing (timing.h): prints the record sizes and, per bounce, the
// average over the iterations so far of the paths entering and leaving it and
// the time of each stage, as "BOUNCE,..." CSV lines. Nothing without the flag.
void pathtraceTimingReport();
