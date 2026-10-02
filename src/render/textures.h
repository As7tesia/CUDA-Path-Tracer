#pragma once

#include <cuda_runtime.h>

class Scene;

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
