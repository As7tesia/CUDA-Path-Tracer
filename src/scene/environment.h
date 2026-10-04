#pragma once

#include <glm/glm.hpp>

#include <string>
#include <vector>

// Where the light from outside the scene comes from: a lat-long image file
// times radiance, or radiance alone in every direction when there is no
// file. A radiance of zero means the scene has no environment.
struct EnvironmentSource
{
    std::string file;  // a .hdr or .exr; empty for one color
    glm::vec3 radiance = glm::vec3(0.0f);
    // Turns the map about +Y before the lookup, in radians: the direction
    // looked up is the ray's direction rotated by this angle, as a Blender
    // Mapping node's Z rotation does to a world texture.
    float rotation = 0.0f;
};

// An EnvironmentSource with its file read: what a path sees in the
// direction it leaves the scene in.
//
// The image is lat-long (equirectangular) with rows top to bottom: the top
// row looks up (+Y), u = 0.5 + atan2(d.z, d.x) / (2 pi), v = acos(d.y) / pi
// for a world direction d. src/render/environment.h reads it on the device.
struct Environment
{
    glm::vec3 radiance = glm::vec3(0.0f);  // multiplies the image, or the color itself
    float rotation = 0.0f;                 // EnvironmentSource::rotation
    int width = 0;
    int height = 0;
    std::vector<float> rgba;               // width * height * 4; empty when there is no image
};

// Reads the source's file, if it has one: Radiance .hdr through stb_image,
// OpenEXR .exr through tinyexr. Throws a std::runtime_error naming the file
// when it cannot be read.
Environment loadEnvironment(const EnvironmentSource& source);

// The folder of environment maps the window lists, and the .hdr and .exr
// files in it, as paths, sorted. Empty when the folder is missing.
extern const std::string ENVIRONMENT_DIR;
std::vector<std::string> environmentMapFiles();
