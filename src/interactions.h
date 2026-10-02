#pragma once

#include "sceneStructs.h"
#include "utilities.h"

#include <glm/glm.hpp>

#include <thrust/random.h>

// Where a ray leaving a surface starts: hitPoint pushed off along normal
// (unit length, pointing to the side the ray leaves on) by EPSILON scaled with
// the hit point's magnitude. A fixed EPSILON is below one float ulp once
// coordinates pass about 170, as in Bistro, and origin + t * dir carries a
// few ulp of error anyway, so the next trace would find the same triangle
// again at t near 0 (both intersection paths accept any t > 0).
__host__ __device__ inline glm::vec3 offsetOrigin(glm::vec3 hitPoint, glm::vec3 normal)
{
    const float scale = fmaxf(1.0f, maxComponent(glm::abs(hitPoint)));
    return hitPoint + normal * (EPSILON * scale);
}

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal, 
    thrust::default_random_engine& rng);

/**
 * Scatter a ray with some probabilities according to the material properties.
 * For example, a diffuse surface scatters in a cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 * In order to apply multiple effects to one surface, probabilistically choose
 * between them.
 *
 * The visual effect you want is to straight-up add the diffuse and specular
 * components. You can do this in a few ways. This logic also applies to
 * combining other types of materias (such as refractive).
 *
 * - Always take an even (50/50) split between a each effect (a diffuse bounce
 *   and a specular bounce), but divide the resulting color of either branch
 *   by its probability (0.5), to counteract the chance (0.5) of the branch
 *   being taken.
 *   - This way is inefficient, but serves as a good starting point - it
 *     converges slowly, especially for pure-diffuse or pure-specular.
 * - Pick the split based on the intensity of each material color, and divide
 *   branch result by that branch's probability (whatever probability you use).
 *
 * This method applies its changes to the Ray parameter `ray` in place.
 * It also modifies the color `color` of the ray in place.
 *
 * You may need to change the parameter list for your purposes!
 */
__host__ __device__ void scatterRay(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    bool outside,
    const Material& m,
    thrust::default_random_engine& rng);
