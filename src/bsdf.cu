#include "bsdf.h"

#include "sampling.h"
#include "utilities.h"

#include <thrust/random.h>

namespace
{
// Below this GGX alpha (roughness 0.01) a lobe is treated as perfectly
// smooth: the microfacet normal is the surface normal and the sample is a
// mirror reflection or a clean refraction.
constexpr float MIN_ALPHA = 1e-4f;

// Where a ray leaving a surface starts: hitPoint pushed off along normal
// (unit length, pointing to the side the ray leaves on) by EPSILON scaled with
// the hit point's magnitude. A fixed EPSILON is below one float ulp once
// coordinates pass about 170, as in Bistro, and origin + t * dir carries a
// few ulp of error anyway, so the next trace would find the same triangle
// again at t near 0 (both intersection paths accept any t > 0).
__host__ __device__ glm::vec3 offsetOrigin(glm::vec3 hitPoint, glm::vec3 normal)
{
    const float scale = fmaxf(1.0f, maxComponent(glm::abs(hitPoint)));
    return hitPoint + normal * (EPSILON * scale);
}

// Smith's Lambda for isotropic GGX, from the cosine between a direction and
// the macro normal (cosTheta > 0).
__host__ __device__ float ggxLambda(float cosTheta, float alpha)
{
    // Floored: a cosine of exactly 0 would make Lambda infinite and G2/G1
    // (1 + inf) / (1 + inf + x), a NaN that would reach the image.
    const float cos2 = fmaxf(cosTheta * cosTheta, 1e-8f);
    const float tan2 = fmaxf(1.0f - cos2, 0.0f) / cos2;
    return 0.5f * (sqrtf(1.0f + alpha * alpha * tan2) - 1.0f);
}

// f * cos / pdf of a GGX lobe whose microfacet normal was drawn from the
// distribution of normals visible from wo, apart from the Fresnel term: the
// height-correlated masking-shadowing G2(wo, wi) over the masking G1(wo)
// (Heitz 2014, 2018). The D terms and the Jacobian of the reflection or
// refraction cancel against the pdf, for reflection and transmission alike.
// Cosines are taken against the macro normal, as absolute values.
__host__ __device__ float smithG2OverG1(float cosO, float cosI, float alpha)
{
    if (alpha < MIN_ALPHA)
    {
        return 1.0f;
    }
    const float lambdaO = ggxLambda(cosO, alpha);
    return (1.0f + lambdaO) / (1.0f + lambdaO + ggxLambda(cosI, alpha));
}

// A microfacet normal from the GGX distribution of normals visible from wo
// (Dupuy and Benyoub 2023, "Sampling Visible GGX Normals with Spherical
// Caps"): stretch wo into the alpha = 1 configuration, sample the spherical
// cap of directions visible from it, unstretch. wo is unit length and on the
// side n faces.
__host__ __device__ glm::vec3 sampleMicrofacetNormal(glm::vec3 n, glm::vec3 wo, float alpha,
    thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);
    const float u1 = u01(rng);
    const float u2 = u01(rng);
    if (alpha < MIN_ALPHA)
    {
        return n;
    }
    glm::vec3 t;
    glm::vec3 b;
    frameAround(n, t, b);
    const glm::vec3 woStd = glm::normalize(glm::vec3(alpha * glm::dot(wo, t), alpha * glm::dot(wo, b), glm::dot(wo, n)));

    const float phi = TWO_PI * u1;
    const float z = (1.0f - u2) * (1.0f + woStd.z) - woStd.z;
    const float sinTheta = sqrtf(glm::clamp(1.0f - z * z, 0.0f, 1.0f));
    const glm::vec3 hStd = glm::vec3(sinTheta * cosf(phi), sinTheta * sinf(phi), z) + woStd;

    const glm::vec3 h = glm::normalize(glm::vec3(alpha * hStd.x, alpha * hStd.y, fmaxf(hStd.z, 1e-6f)));
    return h.x * t + h.y * b + h.z * n;
}

// f0 from the index of refraction against air. glTF's KHR_materials_ior
// reserves ior = 0 for a Fresnel term of 1 everywhere.
__host__ __device__ float iorF0(float ior)
{
    if (ior == 0.0f)
    {
        return 1.0f;
    }
    const float r = (ior - 1.0f) / (ior + 1.0f);
    return r * r;
}
}  // namespace

__host__ __device__ bool scatterPbr(
    PathSegment& path,
    glm::vec3 hitPoint,
    glm::vec3 surfaceNormal,
    bool outside,
    const Material& m,
    const PbrSurface& s,
    thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);
    const glm::vec3 wo = -path.ray.direction;
    const float alpha = s.roughness * s.roughness;

    // Every lobe below either reflects about its normal or transmits through
    // it, and sets wi, the lobe's normal and the weight.
    glm::vec3 wi;
    glm::vec3 lobeNormal;
    glm::vec3 weight;
    bool transmitted = false;

    // A lobe is picked with probability equal to its mixing weight wherever
    // that weight does not depend on wi (clearcoat, metallic, transmission),
    // so the weight cancels. The specular layer's Fresnel weight depends on
    // the microfacet normal, so it is picked with an estimate made from wo
    // and its weight divided by that probability.
    if (hasClearcoat(m, outside)
        && u01(rng) < m.clearcoat * schlickFresnel(glm::vec3(0.04f), glm::dot(s.coatNormal, wo)).x)
    {
        // Clearcoat: a GGX lobe of its own roughness about the mesh normal,
        // on the air side of the surface only (hasClearcoat). fresnel_coat
        // weights it with a Fresnel term at the macro normal, which is the
        // probability it was picked with.
        const float coatAlpha = m.clearcoatRoughness * m.clearcoatRoughness;
        lobeNormal = s.coatNormal;
        const glm::vec3 h = sampleMicrofacetNormal(lobeNormal, wo, coatAlpha, rng);
        wi = glm::reflect(-wo, h);
        weight = glm::vec3(smithG2OverG1(glm::dot(lobeNormal, wo), fabsf(glm::dot(lobeNormal, wi)), coatAlpha));
    }
    else if (u01(rng) < s.metallic)
    {
        // Metal: GGX reflection with Fresnel tinted by the base color.
        lobeNormal = s.normal;
        const glm::vec3 h = sampleMicrofacetNormal(lobeNormal, wo, alpha, rng);
        wi = glm::reflect(-wo, h);
        weight = schlickFresnel(s.baseColor, glm::dot(wo, h))
               * smithG2OverG1(glm::dot(lobeNormal, wo), fabsf(glm::dot(lobeNormal, wi)), alpha);
    }
    else
    {
        // Dielectric. KHR_materials_specular scales the Fresnel term by
        // specularFactor and f0 by specularColorFactor (clamped to 1 before
        // the factor), so the layer reflects specularFactor * fresnel and
        // the base keeps 1 - specularFactor * max(fresnel).
        lobeNormal = s.normal;
        const glm::vec3 f0 = glm::min(iorF0(m.ior) * m.specularColorFactor, glm::vec3(1.0f));
        const float specular = m.specularFactor;

        if (u01(rng) < s.transmission)
        {
            // Glass: one GGX microfacet normal, then reflect or refract
            // through it (Walter et al. 2007) with the probability of the
            // reflected share, so the Fresnel term cancels in both branches.
            const glm::vec3 h = sampleMicrofacetNormal(lobeNormal, wo, alpha, rng);
            const float cosO = glm::dot(wo, h);
            const float eta = outside ? 1.0f / m.ior : m.ior;  // incident over transmitted index
            const float sin2T = eta * eta * (1.0f - cosO * cosO);

            glm::vec3 fresnel(1.0f);  // total internal reflection, or ior = 0
            if (m.ior != 0.0f && sin2T < 1.0f)
            {
                // Schlick takes the cosine on the air side: the incident
                // angle when entering, the transmitted angle when leaving.
                const float cosSchlick = outside ? cosO : sqrtf(1.0f - sin2T);
                fresnel = specular * schlickFresnel(f0, cosSchlick);
            }
            const float pReflect = maxComponent(fresnel);
            const float cosWo = glm::dot(lobeNormal, wo);
            if (u01(rng) < pReflect)
            {
                wi = glm::reflect(-wo, h);
                weight = fresnel / pReflect * smithG2OverG1(cosWo, fabsf(glm::dot(lobeNormal, wi)), alpha);
            }
            else
            {
                // 1 - specular * max(fresnel) over 1 - pReflect is 1.
                wi = glm::refract(-wo, h, eta);
                weight = s.baseColor * smithG2OverG1(cosWo, fabsf(glm::dot(lobeNormal, wi)), alpha);
                transmitted = true;
            }
        }
        else
        {
            // Specular layer over Lambert. Pick the layer by its reflectance
            // seen from wo against the base color. The base's estimate is
            // left unscaled by what the layer takes: at a grazing wo the
            // layer's estimate nears 1 while the Fresnel term at a sampled
            // microfacet can stay small, and 1 / (1 - pLayer) would then
            // turn rare diffuse samples into fireflies. This way a diffuse
            // weight stays below 2.
            const float cosWo = glm::dot(lobeNormal, wo);
            const float layerEstimate = specular * maxComponent(schlickFresnel(f0, cosWo));
            const float baseEstimate = maxComponent(s.baseColor);
            if (layerEstimate + baseEstimate <= 0.0f)
            {
                return false;  // black, nothing to reflect
            }
            const float pLayer = layerEstimate / (layerEstimate + baseEstimate);
            if (u01(rng) < pLayer)
            {
                const glm::vec3 h = sampleMicrofacetNormal(lobeNormal, wo, alpha, rng);
                wi = glm::reflect(-wo, h);
                weight = specular * schlickFresnel(f0, glm::dot(wo, h)) / pLayer
                       * smithG2OverG1(cosWo, fabsf(glm::dot(lobeNormal, wi)), alpha);
            }
            else
            {
                // Lambert: f = color / pi, cosine-weighted pdf = cos / pi.
                wi = sampleCosineHemisphere(lobeNormal, rng);
                const glm::vec3 h = glm::normalize(wo + wi);
                const float layer = specular * maxComponent(schlickFresnel(f0, glm::dot(wo, h)));
                weight = s.baseColor * (1.0f - layer) / (1.0f - pLayer);
            }
        }
    }

    // A reflection has to leave on the lit side of both its lobe's normal
    // and the surface, a transmission on the far side of both. A normal map
    // or a rough lobe can produce directions that fail this; they carry no
    // light (the lobe's masking term is zero there, and past the surface
    // they would leak through it).
    const float side = transmitted ? -1.0f : 1.0f;
    if (side * glm::dot(wi, lobeNormal) <= 0.0f || side * glm::dot(wi, surfaceNormal) <= 0.0f)
    {
        return false;
    }

    path.ray.direction = glm::normalize(wi);
    path.ray.origin = offsetOrigin(hitPoint, side * surfaceNormal);
    path.color *= weight;
    return true;
}
