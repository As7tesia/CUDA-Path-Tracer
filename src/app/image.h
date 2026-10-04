#pragma once

#include <glm/glm.hpp>

#include <string>

class Image
{
private:
    int xSize;
    int ySize;
    glm::vec3 *pixels;

public:
    Image(int x, int y);
    ~Image();
    void setPixel(int x, int y, const glm::vec3 &pixel);
    void savePNG(const std::string &baseFilename);
    // Radiance RGBE: a shared 8-bit exponent and 8-bit mantissas, so values
    // are truncated to steps of 1/256 of their power of two. Fine for
    // looking at, not for means: an image that sits at 1.0 with a little
    // noise loses 0.3% to the truncation. saveEXR is exact.
    void saveHDR(const std::string &baseFilename);
    // OpenEXR with float32 channels, through tinyexr: the pixels as they are.
    void saveEXR(const std::string &baseFilename);
};
