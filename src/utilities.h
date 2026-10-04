#pragma once

#include <cuda_runtime.h>

#include "glm/glm.hpp"

#include <string>

constexpr float PI = 3.1415926535897932384626422832795028841971f;
constexpr float TWO_PI = 6.2831853071795864769252867665590057683943f;
// Ray origin offset off a surface, scaled with the hit point's magnitude (bsdf.cu)
constexpr float EPSILON = 0.00001f;

// The largest of a color's channels: for picking a lobe, or a test for black.
// Callable from kernels; a plain C++ compile of this header sees no attributes.
#ifdef __CUDACC__
__host__ __device__
#endif
inline float maxComponent(glm::vec3 v)
{
    return glm::max(v.x, glm::max(v.y, v.z));
}

// Rec. 709 luminance of a linear color: how bright it looks, for weighting
// lights by power.
#ifdef __CUDACC__
__host__ __device__
#endif
inline float luminance(glm::vec3 c)
{
    return 0.2126f * c.r + 0.7152f * c.g + 0.0722f * c.b;
}

// Handy-dandy hash function that provides seeds for random number generation.
// Here rather than in sampling.h so the OptiX programs, which cannot include
// thrust, can hash too (mesh_hit.h).
#ifdef __CUDACC__
__host__ __device__
#endif
inline unsigned int utilhash(unsigned int a)
{
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

// A file's extension in lower case, with its dot ("Duck.GLB" -> ".glb"),
// or an empty string when it has none.
std::string lowercaseExtension(const std::string& path);

// Prints "error: " and the printf-style message to stderr, then exits with
// EXIT_FAILURE. Every error the program cannot go on from ends here.
[[noreturn]] void fatal(const char* format, ...);

// Ends the program with the error, the call and where it was made when a CUDA
// runtime call fails. checkCUDAError in pathtrace.cu does the same for kernel
// launches.
#define CUDA_CHECK(call) cudaCheck((call), #call, __FILE__, __LINE__)
void cudaCheck(cudaError_t result, const char* call, const char* file, int line);

// The scene JSON's object transform: translation * rotation * scale, the
// rotation Euler angles in degrees, Rx * Ry * Rz (a point turns about z first).
glm::mat4 buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale);
