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
