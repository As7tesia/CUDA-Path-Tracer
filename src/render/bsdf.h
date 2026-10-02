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

// Picks one lobe of the material at a hit and samples a direction from it:
// sets path.ray to leave the hit and multiplies path.color by the lobe's
// f * cos / pdf divided by the probability of picking the lobe. Returns
// false when the sample carries no light (a direction that would pass
// through the surface on the wrong side, or a black material); the path
// should end then, and path is left unchanged.
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
