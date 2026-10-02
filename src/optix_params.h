#pragma once

// Launch parameters for the OptiX intersection stage. Filled per launch by
// optix_intersect.cpp and read on the device by optix_programs.cu, so this
// header is included by both a host .cpp and an OptiX-IR compile: keep it to
// plain structs and pointers.

#include <optix.h>

#include "sceneStructs.h"

// What a hit program needs to know about the instance it hit, indexed by
// optixGetInstanceId() (the Geom's index). A compact record rather than the
// Geom itself, so a hit loads 12 bytes and not three mat4s.
struct InstanceRecord
{
    int materialId;
    int meshId;         // MESH instances: index into MeshBuffers::meshes; -1 otherwise
    float tangentSign;  // Geom::tangentSign
};

struct OptixIntersectParams
{
    const PathSegment* paths;              // paths[i].ray is the ray for launch index i
    ShadeableIntersection* intersections;  // closest-hit and miss write intersections[i]
    int* materialIds;                      // sort key per path: materialId on hit, numMaterials on miss
    int numMaterials;
    const InstanceRecord* instances;       // indexed by instance id
    MeshBuffers buffers;                   // the same flat mesh arrays the naive kernel reads
    // The scene's materials and texture objects, for the any-hit alpha test
    // of ALPHA_MASK materials. textures is null when the scene has none.
    const Material* materials;
    const cudaTextureObject_t* textures;
    OptixTraversableHandle handle;         // the IAS over the unit cube, unit sphere and mesh GASes
};
