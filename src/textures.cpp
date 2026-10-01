// The scene's textures on the GPU. See textures.h.

#include "textures.h"

#include "scene.h"

#include <vector>

namespace
{
std::vector<cudaArray_t> arrays;                // one per Scene::textureImages entry
std::vector<cudaTextureObject_t> objects;       // one per Scene::textures entry
cudaTextureObject_t* deviceObjects = nullptr;   // objects, on the device
}  // namespace

cudaTextureObject_t* texturesInit(const Scene& scene)
{
    if (scene.textures.empty())
    {
        return nullptr;
    }

    const cudaChannelFormatDesc format = cudaCreateChannelDesc<uchar4>();
    for (const TextureImage& image : scene.textureImages)
    {
        cudaArray_t array = nullptr;
        cudaMallocArray(&array, &format, image.width, image.height);
        const size_t rowBytes = (size_t)image.width * 4;
        cudaMemcpy2DToArray(array, 0, 0, image.rgba.data(), rowBytes, rowBytes, image.height, cudaMemcpyHostToDevice);
        arrays.push_back(array);
    }

    for (const Texture& texture : scene.textures)
    {
        cudaResourceDesc resource = {};
        resource.resType = cudaResourceTypeArray;
        resource.res.array.array = arrays[texture.image];

        cudaTextureDesc desc = {};
        desc.addressMode[0] = texture.wrapU;
        desc.addressMode[1] = texture.wrapV;
        desc.filterMode = texture.filter;
        desc.readMode = cudaReadModeNormalizedFloat;  // uchar 0..255 reads as 0..1
        desc.sRGB = texture.srgb ? 1 : 0;
        desc.normalizedCoords = 1;                    // uv in [0, 1] spans the image

        cudaTextureObject_t object = 0;
        cudaCreateTextureObject(&object, &resource, &desc, nullptr);
        objects.push_back(object);
    }

    cudaMalloc(&deviceObjects, objects.size() * sizeof(cudaTextureObject_t));
    cudaMemcpy(deviceObjects, objects.data(), objects.size() * sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice);
    return deviceObjects;
}

void texturesFree()
{
    for (cudaTextureObject_t object : objects)
    {
        cudaDestroyTextureObject(object);
    }
    for (cudaArray_t array : arrays)
    {
        cudaFreeArray(array);
    }
    cudaFree(deviceObjects);  // no-op if null
    objects.clear();
    arrays.clear();
    deviceObjects = nullptr;
}
