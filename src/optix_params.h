#pragma once

// Launch parameters for the OptiX intersection stage. Filled per launch by
// optix_intersect.cpp and read on the device by optix_programs.cu, so this
// header is included by both a host .cpp and an OptiX-IR compile: keep it to
// plain structs and pointers.

#include <optix.h>

#include "sceneStructs.h"

struct OptixIntersectParams
{
    const PathSegment* paths;              // paths[i].ray is the ray for launch index i
    ShadeableIntersection* intersections;  // closest-hit and miss write intersections[i]
    int* materialIds;                      // sort key per path: materialId on hit, numMaterials on miss
    int numMaterials;
    const Geom* geoms;                     // indexed by instance id, for the material id
    OptixTraversableHandle handle;         // the IAS over the unit cube and unit sphere GAS
};
