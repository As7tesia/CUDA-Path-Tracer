#pragma once

// Next event estimation on the device: picking a light from the scene's light
// list (lights.cpp), a point on it, and the MIS weight that splits each
// direction's light between the light sample and the BSDF sample. The shade
// kernel calls these once per hit; the shadow ray that tests the sample is
// traced by the next OptiX launch.

#include <cuda_runtime.h>

#include "render/pbr_surface.h"
#include "scene/sceneStructs.h"
#include "utilities.h"

#include <glm/glm.hpp>

// The scene's light list on the device, uploaded by pathtraceInit.
struct LightList
{
    const LightTriangle* triangles;
    const PunctualLight* punctual;
    const float* cdf;             // running pick probabilities over the triangles, then the punctual lights
    const float* emitterAreaPdf;  // per material, see Scene::emitterAreaPdf
    int numTriangles;
    int numLights;                // triangles plus punctual lights; 0 when there is nothing to sample
};

// The power heuristic with exponent 2 (Veach 1997): the weight of a sample
// drawn with density pThis, when the other strategy would draw the same
// direction with density pOther. Written as a ratio, so a sharp lobe's huge
// density cannot overflow its square: an infinite pThis gives 1 and an
// infinite pOther gives 0. pThis > 0.
__device__ __forceinline__ float powerHeuristic(float pThis, float pOther)
{
    const float r = pOther / pThis;
    return 1.0f / (1.0f + r * r);
}

// The light that u in [0, 1) falls on: the first whose running probability
// is above u, so light i comes up with probability cdf[i] - cdf[i - 1]. A
// binary search, log2(numLights) steps.
__device__ __forceinline__ int pickLight(const LightList& lights, float u)
{
    int lo = 0;
    int hi = lights.numLights - 1;
    while (lo < hi)
    {
        const int mid = (lo + hi) / 2;
        if (lights.cdf[mid] > u)
        {
            hi = mid;
        }
        else
        {
            lo = mid + 1;
        }
    }
    return lo;
}

// The radiance a light triangle's material emits at uv toward a point that
// sees the triangle at cosLight (positive on its front face): pbrEmission's
// value for a ray that hits it there, with the triangle's flat normal for the
// clearcoat's Fresnel term. A cut-out (ALPHA_MASK) emits nothing, and an
// ALPHA_BLEND surface, which a ray hits with probability alpha, emits alpha
// times as much.
__device__ __forceinline__ glm::vec3 lightEmission(const Material& m, glm::vec2 uv, float cosLight,
    const cudaTextureObject_t* textures)
{
    const float4 emissive = sampleSlot(textures, m.emissiveTexture, uv);
    glm::vec3 emission = m.emission * glm::vec3(emissive.x, emissive.y, emissive.z);
    if (hasClearcoat(m, cosLight > 0.0f))
    {
        emission *= 1.0f - m.clearcoat * schlickFresnel(glm::vec3(0.04f), fabsf(cosLight)).x;
    }
    if (m.alphaMode != ALPHA_OPAQUE)
    {
        float alpha = m.alpha;
        if (m.baseColorTexture >= 0)
        {
            alpha *= tex2D<float4>(textures[m.baseColorTexture], uv.x, uv.y).w;
        }
        if (m.alphaMode == ALPHA_MASK)
        {
            return alpha < m.alphaCutoff ? glm::vec3(0.0f) : emission;
        }
        emission *= alpha;
    }
    return emission;
}

// One light sample toward a point.
struct LightSample
{
    glm::vec3 wi;        // unit direction toward the light
    float tMax;          // how far a shadow ray from the point may go without reaching the light itself
    // What arrives at the point divided by the density of sampling it: the
    // emitted radiance over the solid-angle pdf for a triangle, a point
    // light's intensity over d^2 and its pick probability, a distant light's
    // irradiance over its pick probability.
    glm::vec3 weightedLight;
    float pdf;           // solid-angle pdf of wi, for the MIS weight; 0 for a point or distant light
};

// Picks a light with u0 and, for a triangle, a point on it uniformly by area
// with u1 and u2. Returns false when the sample brings no light: the point
// emits nothing there (a dark texel, a cut-out), or the triangle is seen
// edge-on.
//
// A triangle's solid-angle pdf is its area pdf, emitterAreaPdf, times
// d^2 / |cos| of the angle at the light; emitters glow from both sides, so
// the cosine counts either way. A point or distant light is a delta: the BSDF
// can never sample its direction, so it has no pdf to weigh against.
__device__ __forceinline__ bool sampleLight(const LightList& lights, const Material* materials,
    const cudaTextureObject_t* textures, glm::vec3 hitPoint, float u0, float u1, float u2, LightSample& sample)
{
    const int index = pickLight(lights, u0);
    // A shadow ray stops short of the light by a margin the size of the
    // float error at the larger of the two points, or a sliver of the
    // distance, so it does not find the light's own surface.
    const auto margin = [&](glm::vec3 lightPoint, float distance) {
        const float scale = fmaxf(1.0f, fmaxf(maxComponent(glm::abs(hitPoint)), maxComponent(glm::abs(lightPoint))));
        return fmaxf(1e-4f * distance, 4.0f * EPSILON * scale);
    };

    if (index < lights.numTriangles)
    {
        const LightTriangle t = lights.triangles[index];
        // (1 - sqrt(u1), sqrt(u1) u2) as the weights of corners 1 and 2
        // spreads points evenly over the triangle.
        const float s = sqrtf(u1);
        const float b1 = 1.0f - s;
        const float b2 = u2 * s;
        const glm::vec3 p = t.p0 + b1 * t.e1 + b2 * t.e2;
        const glm::vec3 toLight = p - hitPoint;
        const float d2 = glm::dot(toLight, toLight);
        const float d = sqrtf(d2);
        const float tMax = d - margin(p, d);
        const glm::vec3 normal = glm::cross(t.e1, t.e2);
        const float cosLight = -glm::dot(normal, toLight) / (glm::length(normal) * d);
        if (!(tMax > 0.0f) || fabsf(cosLight) < 1e-6f)
        {
            return false;
        }
        const glm::vec2 uv = (1.0f - b1 - b2) * t.uv0 + b1 * t.uv1 + b2 * t.uv2;
        const glm::vec3 emission = lightEmission(materials[t.materialId], uv, cosLight, textures);
        if (emission == glm::vec3(0.0f))
        {
            return false;
        }
        sample.wi = toLight / d;
        sample.tMax = tMax;
        sample.pdf = lights.emitterAreaPdf[t.materialId] * d2 / fabsf(cosLight);
        sample.weightedLight = emission / sample.pdf;
        return true;
    }

    const PunctualLight light = lights.punctual[index - lights.numTriangles];
    sample.pdf = 0.0f;
    if (light.type == LIGHT_POINT)
    {
        const glm::vec3 toLight = light.position - hitPoint;
        const float d2 = glm::dot(toLight, toLight);
        const float d = sqrtf(d2);
        sample.tMax = d - margin(light.position, d);
        if (!(sample.tMax > 0.0f))
        {
            return false;
        }
        sample.wi = toLight / d;
        sample.weightedLight = light.intensity / (d2 * light.pickPdf);
    }
    else
    {
        sample.wi = light.position;
        sample.tMax = 1e16f;  // the path rays' own tmax
        sample.weightedLight = light.intensity / light.pickPdf;
    }
    return true;
}
