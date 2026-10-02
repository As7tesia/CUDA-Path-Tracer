#pragma once

// What both intersection paths compute at a mesh hit, so they agree: the naive
// mesh test (intersections.cu) and the OptiX hit programs (optix_programs.cu).
// Everything here is inline in the header: the OptiX programs are compiled on
// their own to OptiX-IR and cannot call into the other .cu files.

#include <cuda_runtime.h>

#include "scene/sceneStructs.h"
#include "utilities.h"

#include <glm/glm.hpp>

// a[tri.x], a[tri.y] and a[tri.z] weighted by the barycentrics u and v of
// vertices 1 and 2: the one expression both intersection paths interpolate
// vertex attributes with, so they agree bit for bit.
template <typename T>
__device__ __forceinline__ T interpolate(const T* attribute, glm::ivec3 tri, float u, float v)
{
    return (1.0f - u - v) * attribute[tri.x] + u * attribute[tri.y] + v * attribute[tri.z];
}

// The shading normal both intersection paths report for a mesh hit, in world
// space and facing the ray: the vertex normal interpolated at the hit,
// flipped on a back-face hit. Near a silhouette a smooth vertex normal can
// lean past the surface and point away from the ray even on a front-face
// hit; the shade kernel needs a normal that faces the ray, so those hits get
// the geometric normal instead. normalToWorld takes an object-space normal to
// world space (the inverse transpose); each path passes its own, and the
// geometric normal is transformed only when it is used.
template <typename NormalToWorld>
__device__ __forceinline__ glm::vec3 meshShadingNormal(glm::vec3 vertexNormal, glm::vec3 geometricNormal, bool outside,
    glm::vec3 rayDirection, NormalToWorld normalToWorld)
{
    glm::vec3 normal = glm::normalize(normalToWorld(vertexNormal));
    if (!outside)
    {
        normal = -normal;
    }
    if (glm::dot(normal, rayDirection) > 0.0f)
    {
        normal = glm::normalize(normalToWorld(geometricNormal));
        if (!outside)
        {
            normal = -normal;
        }
    }
    return normal;
}

// The tangent both intersection paths report for a mesh hit (see
// ShadeableIntersection::tangent): the vertex tangents interpolated at the
// hit and taken to world space by vectorToWorld, the transform itself, since
// a tangent lies in the surface. A mirroring transform flips the bitangent
// the cross product gives, so tangentSign (Geom::tangentSign) flips the sign
// with it. The sign is the same at all three vertices of a triangle.
template <typename VectorToWorld>
__device__ __forceinline__ glm::vec4 meshTangent(const glm::vec4* tangents, glm::ivec3 tri, float u, float v,
    float tangentSign, VectorToWorld vectorToWorld)
{
    const glm::vec3 t(interpolate(tangents, tri, u, v));
    return glm::vec4(vectorToWorld(t), tangentSign * tangents[tri.x].w);
}

// The seed of one path segment's ALPHA_BLEND tests: a hash of its pixel, the
// iteration and the depth, the triple the shade kernel seeds its engine with.
__device__ __forceinline__ unsigned int alphaPathSeed(int iter, int pixelIndex, int depth)
{
    return utilhash(utilhash(utilhash(pixelIndex) ^ iter) ^ depth);
}

// Whether a hit at uv falls in a cut-out, where the ray goes on as if
// nothing was there: the naive mesh test and the OptiX any-hit program.
// glTF's alpha is the base color factor's alpha times the base color
// texture's; the texture unit reads alpha linearly even from an sRGB texture
// (checked: a texel of 128 reads 0.502 with sRGB on, while its color reads
// 0.216).
//
// ALPHA_MASK cuts out where alpha is below alphaCutoff. ALPHA_BLEND reads
// alpha as the fraction of the surface that is there, so the ray goes
// through with probability 1 - alpha, and the pixel converges to the
// blended result. The number it is decided by is a hash of the path seed
// (alphaPathSeed), the instance (Geom index) and the triangle's index in its
// mesh: both intersection paths hash the same values for the same hit, and
// OptiX, which may run any-hit more than once for a triangle, gets the same
// answer every time.
__device__ __forceinline__ bool alphaCutOut(const Material& m, glm::vec2 uv, const cudaTextureObject_t* textures,
    unsigned int pathSeed, int instance, int triangle)
{
    if (m.alphaMode == ALPHA_OPAQUE)
    {
        return false;
    }
    float alpha = m.alpha;
    if (m.baseColorTexture >= 0)
    {
        alpha *= tex2D<float4>(textures[m.baseColorTexture], uv.x, uv.y).w;
    }
    if (m.alphaMode == ALPHA_MASK)
    {
        return alpha < m.alphaCutoff;
    }
    // The top 24 bits as a float in [0, 1): below 1, so alpha 1 never passes.
    const unsigned int h = utilhash(utilhash(pathSeed ^ instance) ^ triangle);
    return (h >> 8) * (1.0f / 16777216.0f) >= alpha;
}
