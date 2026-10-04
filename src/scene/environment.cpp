// Environment map files. See environment.h.
//
// This is the one translation unit that compiles tinyexr's implementation
// (v1.0.13, external/include with its exr_reader.hh and streamreader.hh). Its
// ZIP decoding uses the zlib inside the stb headers that stb.cpp compiles,
// so no miniz is vendored; stb.cpp gives stbi_zlib_compress the C linkage
// tinyexr expects. Chunks decode on several threads.

#define TINYEXR_USE_MINIZ 0
#define TINYEXR_USE_STB_ZLIB 1
#define TINYEXR_USE_THREAD 1
#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"

#include "scene/environment.h"

#include "timing.h"
#include "utilities.h"

#include <stb_image.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>

const std::string ENVIRONMENT_DIR = "scenes/assets/hdri/";

namespace
{
// Reads a lat-long HDR image into env's image.
void loadEnvironmentImage(const std::string& path, Environment& env)
{
    const std::string ext = lowercaseExtension(path);
    float* pixels = nullptr;
    int width = 0;
    int height = 0;
    if (ext == ".exr")
    {
        // LoadEXR returns RGBA floats, alpha 1 when the file has none.
        const char* err = nullptr;
        if (LoadEXR(&pixels, &width, &height, path.c_str(), &err) != TINYEXR_SUCCESS)
        {
            const std::string reason = err != nullptr ? err : "unknown error";
            FreeEXRErrorMessage(err);
            throw std::runtime_error(path + ": " + reason);
        }
    }
    else if (ext == ".hdr")
    {
        // stbi_loadf would also read an 8-bit image, through an assumed
        // gamma of 2.2, so only Radiance files come this way.
        int channels = 0;
        pixels = stbi_loadf(path.c_str(), &width, &height, &channels, 4);
        if (pixels == nullptr)
        {
            throw std::runtime_error(path + ": " + stbi_failure_reason());
        }
    }
    else
    {
        throw std::runtime_error(path + ": not an environment map (.hdr or .exr)");
    }

    env.width = width;
    env.height = height;
    env.rgba.assign(pixels, pixels + (size_t)width * height * 4);
    free(pixels);  // both libraries allocate with malloc
    printf("Environment %s: %dx%d\n", path.c_str(), width, height);
}
}  // namespace

Environment loadEnvironment(const EnvironmentSource& source)
{
    TimingScope timing("load.environment");
    Environment env;
    env.radiance = source.radiance;
    env.rotation = source.rotation;
    if (!source.file.empty())
    {
        loadEnvironmentImage(source.file, env);
    }
    return env;
}

std::vector<std::string> environmentMapFiles()
{
    std::vector<std::string> files;
    if (!std::filesystem::is_directory(ENVIRONMENT_DIR))
    {
        return files;
    }
    for (const auto& entry : std::filesystem::directory_iterator(ENVIRONMENT_DIR))
    {
        const std::string ext = lowercaseExtension(entry.path().string());
        if (entry.is_regular_file() && (ext == ".hdr" || ext == ".exr"))
        {
            files.push_back(entry.path().generic_string());
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}
