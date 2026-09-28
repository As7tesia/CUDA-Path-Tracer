#pragma once

#include "sceneStructs.h"

#include <cuda_runtime.h>

class Scene;

// OptiX intersection stage: the same contract as computeIntersections in
// pathtrace.cu (intersections[i].t = -1 on a miss, materialIds[i] = materialId
// on a hit or numMaterials on a miss) with traversal on the RT cores. The host
// side lives in optix_intersect.cpp, a .cpp and not a .cu: optix.h switches to
// the device API whenever __CUDACC__ is defined, so host code that includes
// it cannot be compiled by nvcc. pathtrace.cu only sees this header.

// Creates the OptiX context, builds the acceleration structures for the
// scene's geometry, the pipeline and the shader binding table. Called once
// from pathtraceInit, after it uploaded the scene's flat mesh arrays: the
// mesh acceleration structures are built from those device buffers and the
// hit program reads vertex data from them, so they must outlive this stage.
// Returns false, after printing why, if the driver has no OptiX or any step
// fails; the caller then keeps the naive kernel. validation turns on OptiX
// validation mode, which checks every launch and is slow.
bool optixIntersectInit(const Scene* scene, const MeshBuffers& buffers, bool validation);
void optixIntersectFree();

// Traces paths[0, numPaths) and writes intersections and materialIds for each.
// The buffers may change between calls (they ping-pong in the wavefront loop).
void optixIntersect(int numPaths, const PathSegment* paths,
                    ShadeableIntersection* intersections, int* materialIds,
                    int numMaterials, cudaStream_t stream = 0);
