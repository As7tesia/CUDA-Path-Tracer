#pragma once

#include "scene/sceneStructs.h"

#include <glm/glm.hpp>


/**
 * Compute a point at parameter value `t` on ray `r`. Exactly on the surface;
 * scatterPbr offsets the next ray's origin along the normal instead.
 */
__host__ __device__ inline glm::vec3 getPointOnRay(Ray r, float t)
{
    return r.origin + t * glm::normalize(r.direction);
}

/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ inline glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v)
{
    return glm::vec3(m * v);
}

/**
 * Test intersection between a ray and a transformed cube. Untransformed,
 * the cube ranges from -0.5 to 0.5 in each axis and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside);

/**
 * Test intersection between a ray and a transformed sphere. Untransformed,
 * the sphere always has radius 0.5 and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside);

/**
 * Test intersection between a ray and one instance of a triangle mesh: every
 * triangle of `mesh` under geom's transform, closest hit wins. The normal is
 * the vertex normal interpolated at the hit, flipped to face the ray when the
 * ray hit the back face, like sphereIntersectionTest. `outside` is whether
 * the ray hit the front face (counterclockwise winding seen from the ray),
 * which for a closed mesh means the ray came from outside. `uv` is the vertex
 * uvs interpolated at the hit, `tangent` the vertex tangents in world space
 * (see ShadeableIntersection::tangent). Hits in the cut-outs of an
 * ALPHA_MASK material do not count, the same test OptiX's any-hit program
 * makes.
 *
 * Device only: the alpha test reads textures.
 *
 * @return  Ray parameter `t` value. -1 if no intersection.
 */
__device__ float meshIntersectionTest(
    const Geom& geom,
    const TriangleMesh& mesh,
    const MeshBuffers& buffers,
    const Material& material,
    const cudaTextureObject_t* textures,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    glm::vec2& uv,
    glm::vec4& tangent,
    bool& outside);
