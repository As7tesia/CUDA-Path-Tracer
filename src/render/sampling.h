#pragma once

// Random numbers and the sampling helpers built on them, for the device and
// for the host test of the BSDF.

#include "utilities.h"

#include <glm/glm.hpp>
#include <thrust/random.h>

// A random number engine for one path at one depth of one iteration, seeded
// from a hash of the three, so every such triple draws its own numbers and a
// render repeats bit for bit.
__host__ __device__ inline thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1u << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

// Unit vectors t, b so that (t, b, n) is an orthonormal frame, for unit n
// (Duff et al. 2017, "Building an Orthonormal Basis, Revisited"). The BSDF
// is isotropic, so any frame around n works.
__host__ __device__ inline void frameAround(glm::vec3 n, glm::vec3& t, glm::vec3& b)
{
    const float sign = copysignf(1.0f, n.z);
    const float a = -1.0f / (sign + n.z);
    const float c = n.x * n.y * a;
    t = glm::vec3(1.0f + sign * n.x * n.x * a, sign * c, -sign * n.x);
    b = glm::vec3(c, sign + n.y * n.y * a, -n.y);
}

// A direction in the hemisphere around unit n with density cos(theta) / pi,
// the Lambert lobe's: cos(theta) = sqrt(u1) and a uniform angle around n.
__host__ __device__ inline glm::vec3 sampleCosineHemisphere(glm::vec3 n, thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);
    const float cosTheta = sqrtf(u01(rng));
    const float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);
    const float phi = TWO_PI * u01(rng);
    glm::vec3 t;
    glm::vec3 b;
    frameAround(n, t, b);
    return cosTheta * n + (sinTheta * cosf(phi)) * t + (sinTheta * sinf(phi)) * b;
}
