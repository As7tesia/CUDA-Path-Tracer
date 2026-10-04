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
//
// The table is what next event estimation samples the map with
// (environmentSample in src/render/environment.h): a density over the (u, v)
// square that is constant inside each texel's cell and follows the map's
// luminance times sin(theta), so a texel gets picks in proportion to the
// light it adds to the scene. It is drawn in two 1D steps, a row by the
// marginal CDF and a column by that row's conditional CDF. A sample's
// radiance is then read from the texture, which filters bilinearly, so the
// value it returns inside a cell slides from the texel's own value toward
// the neighbors' at the edges. The cell's luminance in the table is the
// average of that filtered read over the cell, the [1 6 1]/8 filter of the
// texels in u (wrapping) and in v (clamped), so the picks follow what the
// texture returns rather than the one value stored at the center. (With MIS
// this is a matter of variance, not bias: a direction the table never
// draws is still covered by the BSDF sample at weight 1.) With MIS
// compensation (Karlik et al. 2019) the map's average
// radiance is subtracted first and the rest clamped at zero: the even part
// of the sky, which BSDF sampling handles on its own, gets no picks, and
// they all go to the sun and the bright patches.
struct Environment
{
    glm::vec3 radiance = glm::vec3(0.0f);  // multiplies the image, or the color itself
    float rotation = 0.0f;                 // EnvironmentSource::rotation
    int width = 0;
    int height = 0;
    std::vector<float> rgba;               // width * height * 4; empty when there is no image

    // The map's luminance integrated over all directions, in W/m^2 (4 pi
    // times the average luminance). The light list weighs the environment
    // by what it pours onto the scene's bounding disk, pi r^2 times this.
    float radianceIntegral = 0.0f;
    // The sampling table; empty when there is no image (a one-color
    // environment is sampled uniformly). pdfUv has the density of each
    // cell over the unit square (mean 1 over the cells), conditionalCdf a
    // running sum per row over its columns ending at 1, marginalCdf the
    // running sum over the rows. The density over directions is
    // pdfUv / (2 pi^2 sin(theta)).
    std::vector<float> pdfUv;           // width * height
    std::vector<float> conditionalCdf;  // width * height
    std::vector<float> marginalCdf;     // height
    bool compensated = false;           // the table had the average subtracted
};

// Reads the source's file, if it has one: Radiance .hdr through stb_image,
// OpenEXR .exr through tinyexr, and builds its sampling table, with MIS
// compensation when compensate is set. Throws a std::runtime_error naming
// the file when it cannot be read.
Environment loadEnvironment(const EnvironmentSource& source, bool compensate);

// The folder of environment maps the window lists, and the .hdr and .exr
// files in it, as paths, sorted. Empty when the folder is missing.
extern const std::string ENVIRONMENT_DIR;
std::vector<std::string> environmentMapFiles();
