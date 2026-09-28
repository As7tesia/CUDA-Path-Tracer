#pragma once

#include <cuda_runtime.h>

#include "glm/glm.hpp"

#include <string>
#include <vector>

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType
{
    SPHERE,
    CUBE,
    MESH  // one glTF primitive's triangles, see TriangleMesh
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

// One glTF primitive: triCount consecutive triangles of the scene's flat
// index array. Each index points into the flat vertex arrays (positions and
// normals share it), already offset, so a triangle needs no base vertex. A
// Geom of type MESH instances one of these under its transform; several
// Geoms may share one (a mesh used by more than one glTF node).
struct TriangleMesh
{
    int indexOffset;  // first triangle in the index array
    int triCount;
};

// Device pointers to the scene's flat mesh arrays, uploaded once by
// pathtraceInit and read by both intersection paths (the naive kernel loops
// over them; OptiX builds its acceleration structures from them and reads the
// normals in its hit program). Read-only after the upload.
struct MeshBuffers
{
    glm::vec3* positions;
    glm::vec3* normals;     // per vertex, unit length, object space
    glm::ivec3* indices;    // per triangle, into positions / normals
    TriangleMesh* meshes;   // indexed by Geom::meshId
};

struct Geom
{
    enum GeomType type;
    int materialid;
    int meshId;  // MESH only: which TriangleMesh; -1 for the primitives
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
};

enum MaterialType
{
    DIFFUSE,
    SPECULAR,
    REFRACTIVE,
    EMISSIVE
};

struct Material
{
    MaterialType type;
    glm::vec3 color;
    struct
    {
        float exponent;
        glm::vec3 color;
    } specular;
    float ior;
    float emittance;
};

struct Camera
{
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
};

struct RenderState
{
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment
{
    Ray ray;
    glm::vec3 color;
    int pixelIndex;
    int remainingBounces;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
  float t;
  glm::vec3 surfaceNormal;
  int materialId;
  bool outside;
};
