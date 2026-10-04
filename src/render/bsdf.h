#pragma once

// The BSDF of every material: glTF's metallic-roughness model (Appendix B of
// the glTF 2.0 spec plus the extensions this project reads), sampled one lobe
// at a time:
//
//   coated   = fresnel_coat(base = material, layer = GGX(clearcoatRoughness^2))  KHR_materials_clearcoat
//   material = mix(dielectric, metal, metallic)
//   metal    = GGX(roughness^2) with Schlick Fresnel, f0 = base color
//   dielectric = fresnel_mix(f0 from ior and KHR_materials_specular,
//                  layer = GGX(roughness^2),
//                  base  = mix(Lambert(base color), GGX transmission tinted by base color, transmission))
//
// One roughness serves reflection and transmission, and the transmission is
// a solid boundary: it refracts with ior (KHR_materials_ior) on the way in and
// out whether or not the material has KHR_materials_volume, which only adds
// absorption (applied by the shade kernel).

#include "render/pbr_surface.h"
#include "scene/sceneStructs.h"

#include <glm/glm.hpp>
#include <thrust/random.h>

// Where a ray leaving a surface starts: hitPoint pushed off along normal
// (unit length, pointing to the side the ray leaves on) by EPSILON scaled with
// the hit point's magnitude. A fixed EPSILON is below one float ulp once
// coordinates pass about 170, as in Bistro, and origin + t * dir carries a
// few ulp of error anyway, so the next trace would find the same triangle
// again at t near 0 (both intersection paths accept any t > 0). Shadow rays
// start the same way.
__host__ __device__ glm::vec3 offsetOrigin(glm::vec3 hitPoint, glm::vec3 normal);

// Picks one lobe of the material at a hit and samples a direction from it:
// sets path.ray to leave the hit and multiplies path.color by the lobe's
// f * cos / pdf divided by the probability of picking the lobe. Sets
// path.pdf to evalPbr's pdf of the new direction, or 0 when a smooth lobe
// made it. Returns false when the sample carries no light (a direction that
// would pass through the surface on the wrong side, or a black material);
// the path should end then, and path is left unchanged.
//
// hitPoint is on the surface, surfaceNormal is the unmapped normal facing the
// incoming ray (it decides which side a new ray starts on), outside says
// whether the ray hit the front face (it decides the ior ratio).
__host__ __device__ bool scatterPbr(
    PathSegment& path,
    glm::vec3 hitPoint,
    glm::vec3 surfaceNormal,
    bool outside,
    const Material& m,
    const PbrSurface& s,
    thrust::default_random_engine& rng);

// The BSDF at a direction someone else picked, for next event estimation.
struct BsdfEval
{
    glm::vec3 fCos;  // f(wo, wi) * |cos| of wi against each lobe's normal, summed over the lobes
    float pdf;       // the solid-angle density scatterPbr picks wi with
};

// f * cos and pdf at (wo, wi) over the lobes that are not smooth, each
// weighted by the probability scatterPbr picks it with: those probabilities
// depend on wo alone, so the mixture's pdf is exact. A smooth lobe (GGX
// alpha below the cutoff in bsdf.cu, or a transmission with ior 1) reflects
// or refracts into one direction only, which a light sample never lands on,
// so it adds nothing to either. Returns zeros where scatterPbr would reject
// wi: a reflection has to leave on the lit side of both its lobe's normal
// and surfaceNormal, a transmission on the far side of both.
//
// wo points from the hit back along the incoming ray, wi away from the hit,
// both unit length. The other arguments are scatterPbr's.
__host__ __device__ BsdfEval evalPbr(
    glm::vec3 wo,
    glm::vec3 wi,
    glm::vec3 surfaceNormal,
    bool outside,
    const Material& m,
    const PbrSurface& s);
