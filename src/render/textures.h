#pragma once

#include <cuda_runtime.h>

#include <glm/glm.hpp>

class Scene;
struct Environment;

// The scene's textures on the GPU. Each Scene::textureImages entry becomes a
// CUDA array of uchar4, and each Scene::textures entry a texture object over
// its image's array: normalized coordinates, the sampler's wrap and filter
// modes, read as floats in [0, 1], with the texture unit decoding sRGB to
// linear for color slots. Host code only (no kernels), so pathtrace.cu keeps
// its compile time; the lookup itself is a tex2D in the shade kernel.

// Uploads the images, creates the texture objects and returns a device array
// of them indexed like Scene::textures, or null when the scene has none.
// Called once from pathtraceInit.
cudaTextureObject_t* texturesInit(const Scene& scene);

// Destroys the texture objects and frees the arrays and the device array.
void texturesFree();

// The scene's Environment as the shade kernel reads it, passed by value.
// environmentRadiance (render/environment.h) does the lookup.
struct EnvironmentMap
{
    // A float4 CUDA array of the lat-long image, filtered bilinearly,
    // wrapping around the horizon and clamped at the poles; 0 when the
    // environment is one color
    cudaTextureObject_t texture;
    glm::vec3 radiance;  // multiplies the image, or the color itself; zero: no environment
    float cosRotation;   // Environment::rotation
    float sinRotation;
    // The sampling table, Environment's three arrays (scene/environment.h),
    // for environmentSample and environmentPdf; null when the environment
    // is one color
    const float* pdfUv;           // width * height
    const float* conditionalCdf;  // width * height
    const float* marginalCdf;     // height
    int width;
    int height;
};

// Uploads the environment's image, if it has one. Called once from
// pathtraceInit.
EnvironmentMap environmentInit(const Environment& env);

// Destroys the environment's texture object and frees its array.
void environmentFree();
