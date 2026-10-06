#pragma once

// Next event estimation on the device: picking a light from the scene's light
// list (lights.cpp), a point on it, and the MIS weight that splits each
// direction's light between the light sample and the BSDF sample. The shade
// kernel calls these once per hit; the shadow ray that tests the sample is
// traced by the next OptiX launch.

#include <cuda_runtime.h>

#include "render/environment.h"
#include "render/pbr_surface.h"
#include "scene/sceneStructs.h"
#include "utilities.h"

#include <glm/glm.hpp>

// The scene's light list on the device, uploaded by pathtraceInit.
struct LightList
{
    const LightTriangle* triangles;
    const PunctualLight* punctual;
    // The pick tables, one row per receiver mask (Scene::lightMasks): the
    // running pick probabilities over the triangles, the punctual lights,
    // then the environment; per material, Scene::emitterAreaPdf; per
    // punctual light, its pick probability; and the environment's, 0 when
    // the scene has no environment to sample. A row counts only the lights
    // its mask receives. Row m starts at m * numLights, m * numMaterials,
    // m * numPunctual, and m.
    const float* cdf;
    const float* emitterAreaPdf;
    const float* punctualPickPdf;
    const float* environmentPickPdf;
    const unsigned int* masks;    // the distinct masks, all ones first
    int numMasks;
    int numMaterials;
    int numPunctual;
    int numTriangles;
    int numLights;                // triangles plus punctual lights plus the environment; 0 when there is nothing to sample
};

// The row of the pick tables for a surface's mask. All ones is row 0, and
// stands in for a mask the tables do not hold (a camera ray's, or one past
// the table's limit): its picks of lights the surface does not receive are
// zeros, so the estimate stays right and only samples are lost.
__device__ __forceinline__ int lightRow(const LightList& lights, unsigned int mask)
{
    for (int m = 1; m < lights.numMasks; ++m)
    {
        if (lights.masks[m] == mask)
        {
            return m;
        }
    }
    return 0;
}

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

// Whether a surface with receiverMask (Geom::lightMask) receives light from
// link group group.
__device__ __forceinline__ bool receivesGroup(unsigned int receiverMask, int group)
{
    return ((receiverMask >> group) & 1u) != 0u;
}

// A spot light's falloff toward a point it sees in direction -wi (wi points
// from the point to the light): 1 inside the inner cone, 0 outside the outer
// one, and the smoothstep 3t^2 - 2t^3 of the cosine between, Cycles' spot
// falloff when the exporter maps its blend to the inner angle.
__device__ __forceinline__ float spotFalloff(const PunctualLight& light, glm::vec3 wi)
{
    const float cosAngle = -glm::dot(wi, light.direction);
    if (cosAngle <= light.cosOuter)
    {
        return 0.0f;
    }
    if (cosAngle >= light.cosInner)
    {
        return 1.0f;
    }
    const float t = (cosAngle - light.cosOuter) / (light.cosInner - light.cosOuter);
    return t * t * (3.0f - 2.0f * t);
}

// One light sample toward a point.
struct LightSample
{
    glm::vec3 wi;        // unit direction toward the light
    float tMax;          // how far a shadow ray from the point may go without reaching the light itself
    // What arrives at the point divided by the density of sampling it: the
    // emitted radiance over the solid-angle pdf for a triangle, a point
    // light's intensity over d^2 and its pick probability, a distant light's
    // irradiance over its pick probability, the environment's radiance over
    // its solid-angle pdf.
    glm::vec3 weightedLight;
    float pdf;           // solid-angle pdf of wi, for the MIS weight; 0 for a point or distant light
};

// Picks a light with u0 and, for a triangle, a point on it uniformly by area
// with u1 and u2, or for the environment a direction from its table.
// Returns false when the sample brings no light: the point emits nothing
// there (a dark texel, a cut-out), the triangle is seen edge-on, the point
// lies outside a spot light's cone, or the surface at hitPoint does not
// receive the light's link group (receiverMask, see Geom::lightMask). The
// pick table row of receiverMask gives such a light no share, so the last
// case only happens for a mask without a row of its own; a dropped sample
// is a zero either way, which keeps the estimate unbiased.
//
// A triangle's solid-angle pdf is its area pdf, emitterAreaPdf, times
// d^2 / |cos| of the angle at the light; emitters glow from both sides, so
// the cosine counts either way. The environment's is its pick probability
// times environmentPdf. A point or distant light is a delta: the BSDF can
// never sample its direction, so it has no pdf to weigh against.
__device__ __forceinline__ bool sampleLight(const LightList& lights, const EnvironmentMap& env,
    const Material* materials, const cudaTextureObject_t* textures, glm::vec3 hitPoint, unsigned int receiverMask,
    float u0, float u1, float u2, LightSample& sample)
{
    // The light u0 falls on: light i with probability cdf[i] - cdf[i - 1]
    // of the receiver's row
    const int row = lightRow(lights, receiverMask);
    const int index = searchCdf(lights.cdf + row * lights.numLights, lights.numLights, u0);
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
        if (!receivesGroup(receiverMask, t.group))
        {
            return false;
        }
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
        sample.pdf = lights.emitterAreaPdf[row * lights.numMaterials + t.materialId] * d2 / fabsf(cosLight);
        sample.weightedLight = emission / sample.pdf;
        return true;
    }

    // The environment: a direction from its table, infinitely far like a
    // distant light, so the shadow ray runs to the path rays' own tmax.
    const float environmentPick = lights.environmentPickPdf[row];
    if (environmentPick > 0.0f && index == lights.numLights - 1)
    {
        float pdf;
        if (!environmentSample(env, u1, u2, sample.wi, pdf))
        {
            return false;
        }
        sample.tMax = 1e16f;
        sample.pdf = environmentPick * pdf;
        sample.weightedLight = environmentRadiance(env, sample.wi) / sample.pdf;
        return true;
    }

    const PunctualLight light = lights.punctual[index - lights.numTriangles];
    if (!receivesGroup(receiverMask, light.group))
    {
        return false;
    }
    const float pickPdf = lights.punctualPickPdf[row * lights.numPunctual + index - lights.numTriangles];
    sample.pdf = 0.0f;
    if (light.type != LIGHT_DISTANT)
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
        sample.weightedLight = light.intensity / (d2 * pickPdf);
        if (light.type == LIGHT_SPOT)
        {
            const float falloff = spotFalloff(light, sample.wi);
            if (falloff <= 0.0f)
            {
                return false;
            }
            sample.weightedLight *= falloff;
        }
    }
    else
    {
        sample.wi = light.position;
        sample.tMax = 1e16f;  // the path rays' own tmax
        sample.weightedLight = light.intensity / pickPdf;
    }
    return true;
}
