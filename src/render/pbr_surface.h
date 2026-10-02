#pragma once

// A material at one hit, read on the device: its texture lookups, the
// normal map, and the Fresnel and clearcoat rules the emission shares with
// the BSDF (bsdf.cu). The shade kernel calls pbrSurface or pbrEmission once
// per hit.

#include <cuda_runtime.h>

#include "scene/sceneStructs.h"

#include <glm/glm.hpp>

// A material at one hit after its texture lookups: the inputs scatterPbr
// (bsdf.h) samples and the emission the shade kernel adds.
struct PbrSurface
{
    glm::vec3 baseColor;
    float metallic;
    float roughness;
    float transmission;
    glm::vec3 emission;
    glm::vec3 normal;      // shading normal: normal-mapped, facing the incoming ray
    glm::vec3 coatNormal;  // the clearcoat's: the mesh normal, which the normal map does not touch
};

// A texture slot's value at uv, or 1 in every channel when the slot is
// empty, so the factor the texture multiplies stands alone (glTF reads a
// missing texture as 1).
__device__ __forceinline__ float4 sampleSlot(const cudaTextureObject_t* textures, int slot, glm::vec2 uv)
{
    return slot >= 0 ? tex2D<float4>(textures[slot], uv.x, uv.y) : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
}

// The normal from a normal map texel, in world space and facing the ray like
// isect.surfaceNormal. The tangent frame is built on the front-face normal,
// the side the map was authored for, and the result is flipped for a
// back-face hit. The tangent is made perpendicular to the normal first
// (interpolation and non-uniform scales tilt it). Falls back to the unmapped
// normal when the tangent is degenerate, or when the mapped normal faces away
// from the ray (a steep map seen at a grazing angle), where no direction
// would be on the lit side of both it and the surface.
__device__ __forceinline__ glm::vec3 mappedNormal(const ShadeableIntersection& isect, float4 texel, float scale,
    glm::vec3 wo)
{
    const glm::vec3 n = isect.outside ? isect.surfaceNormal : -isect.surfaceNormal;
    const glm::vec3 tangent(isect.tangent);
    glm::vec3 t = tangent - n * glm::dot(n, tangent);
    const float length2 = glm::dot(t, t);
    if (length2 < 1e-12f)
    {
        return isect.surfaceNormal;
    }
    t *= rsqrtf(length2);
    const glm::vec3 b = isect.tangent.w * glm::cross(n, t);

    // Texels map [0, 1] onto [-1, 1]; normalTexture.scale scales x and y.
    const float x = (2.0f * texel.x - 1.0f) * scale;
    const float y = (2.0f * texel.y - 1.0f) * scale;
    const float z = 2.0f * texel.z - 1.0f;
    glm::vec3 mapped = glm::normalize(x * t + y * b + z * n);
    if (!isect.outside)
    {
        mapped = -mapped;
    }
    return glm::dot(mapped, wo) > 0.0f ? mapped : isect.surfaceNormal;
}

// Whether the clearcoat applies at a hit. The coat lies on the air side of
// the surface: inside a transmissive material (a back-face hit, the inside of
// a solid) the coat and the glass share an index and nothing reflects, so
// neither the coat lobe nor its darkening of the emission applies there. A
// back face of an opaque material is a double-sided surface seen from behind,
// which keeps its coat. The transmission factor decides, as a transmission
// texture does not change which side is the inside.
__host__ __device__ __forceinline__ bool hasClearcoat(const Material& m, bool outside)
{
    return m.clearcoat > 0.0f && (outside || m.transmission == 0.0f);
}

// Schlick's Fresnel approximation with f90 = 1.
__host__ __device__ __forceinline__ glm::vec3 schlickFresnel(glm::vec3 f0, float cosTheta)
{
    const float m = glm::clamp(1.0f - cosTheta, 0.0f, 1.0f);
    const float m2 = m * m;
    return f0 + (glm::vec3(1.0f) - f0) * (m2 * m2 * m);
}

// What a material emits at a hit: the emissive factor times its texture,
// darkened by the clearcoat above it (KHR_materials_clearcoat puts the coat
// above the emission, so the light the coat reflects is light the surface
// does not emit). On its own for a hit the path ends at, which needs nothing
// else of the material; pbrSurface calls it for the rest.
__device__ __forceinline__ glm::vec3 pbrEmission(const Material& m, const ShadeableIntersection& isect, glm::vec3 wo,
    const cudaTextureObject_t* textures)
{
    const float4 emissive = sampleSlot(textures, m.emissiveTexture, isect.uv);
    glm::vec3 emission = m.emission * glm::vec3(emissive.x, emissive.y, emissive.z);
    if (hasClearcoat(m, isect.outside))
    {
        const float coatFresnel = schlickFresnel(glm::vec3(0.04f), glm::dot(isect.surfaceNormal, wo)).x;
        emission *= 1.0f - m.clearcoat * coatFresnel;
    }
    return emission;
}

// A material's inputs at a hit. wo points from the hit back along the
// incoming ray. Texture values multiply their factors; metallic, roughness
// and transmission are clamped to [0, 1] as glTF defines them.
__device__ __forceinline__ PbrSurface pbrSurface(const Material& m, const ShadeableIntersection& isect, glm::vec3 wo,
    const cudaTextureObject_t* textures)
{
    const glm::vec2 uv = isect.uv;
    PbrSurface s;

    const float4 base = sampleSlot(textures, m.baseColorTexture, uv);
    s.baseColor = m.baseColor * glm::vec3(base.x, base.y, base.z);

    // glTF packs roughness in G and metallic in B (R is free for occlusion).
    const float4 mr = sampleSlot(textures, m.metallicRoughnessTexture, uv);
    s.roughness = glm::clamp(m.roughness * mr.y, 0.0f, 1.0f);
    s.metallic = glm::clamp(m.metallic * mr.z, 0.0f, 1.0f);
    s.transmission = glm::clamp(m.transmission * sampleSlot(textures, m.transmissionTexture, uv).x, 0.0f, 1.0f);

    s.emission = pbrEmission(m, isect, wo, textures);

    s.coatNormal = isect.surfaceNormal;
    s.normal = isect.surfaceNormal;
    if (m.normalTexture >= 0 && isect.tangent.w != 0.0f)
    {
        s.normal = mappedNormal(isect, tex2D<float4>(textures[m.normalTexture], uv.x, uv.y), m.normalScale, wo);
    }
    return s;
}
