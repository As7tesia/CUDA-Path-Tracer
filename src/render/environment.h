#pragma once

// The environment lookup on the device, for kernels only (tex2D). The
// lat-long layout is Environment's (scene/environment.h).

#include <cuda_runtime.h>

#include "render/textures.h"
#include "utilities.h"

#include <glm/glm.hpp>

// The light arriving from outside the scene along a unit world direction.
// The direction is first turned about +Y by the scene's rotation, the
// rotation matrix [c 0 s; 0 1 0; -s 0 c].
__device__ __forceinline__ glm::vec3 environmentRadiance(const EnvironmentMap& env, glm::vec3 direction)
{
    if (env.texture == 0)
    {
        return env.radiance;
    }
    const float c = env.cosRotation;
    const float s = env.sinRotation;
    const glm::vec3 d(c * direction.x + s * direction.z, direction.y, -s * direction.x + c * direction.z);
    const float u = 0.5f + atan2f(d.z, d.x) * (1.0f / TWO_PI);
    const float v = acosf(glm::clamp(d.y, -1.0f, 1.0f)) * (1.0f / PI);
    const float4 texel = tex2D<float4>(env.texture, u, v);
    return env.radiance * glm::vec3(texel.x, texel.y, texel.z);
}
