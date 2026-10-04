#pragma once

#include <glm/glm.hpp>

#include <optional>
#include <string>

class Scene;

// A perspective camera from a glTF file, in world space. A glTF camera looks
// down its node's -Z axis with +Y up, so the pose is read off the node's
// world transform. znear and zfar are not kept: rays are not clipped.
struct GltfCamera
{
    glm::vec3 position;
    glm::vec3 view;     // unit length
    glm::vec3 up;       // unit length
    bool mirrored;      // the transform flips handedness, see Camera::mirrored
    float yfov;         // full vertical field of view, radians
    float aspectRatio;  // width / height, 0 when the file leaves it to the viewport
};

// Render settings. glTF has none; the glTF research scenes
// (github.com/ErfanMo77/gltf-research-scenes) carry their PBRT source's
// settings in the root extras, under pbrt.render. 0 = not in the file.
struct GltfRenderHints
{
    int width = 0;
    int height = 0;
    int maxDepth = 0;
};

// A research scene's PBRT infinite light, from the root extras under
// pbrt.light_sources: an image map, or one radiance in every direction.
// PBRT's maps are equal-area octahedral squares and the renderer reads
// lat-long maps only, so tools/convert_envmaps.py converts each map once,
// with the light's rotation baked in, to <map>.latlong.exr next to it.
struct GltfEnvironment
{
    std::string latlongFile;  // the converted map; empty for a constant light
    std::string pbrtFile;     // the PBRT map it is made from
    glm::vec3 radiance;       // radiance_rgb of a constant light, 1 for a map
};

// What a file holds besides the geometry and materials that went into the scene.
struct GltfInfo
{
    // World-space bounds of the geometry added. min > max when there is none.
    glm::vec3 boundsMin;
    glm::vec3 boundsMax;
    // The first perspective camera in the node walk's order (depth first from
    // the default scene's roots), if the file has one.
    std::optional<GltfCamera> camera;
    GltfRenderHints render;
    // The first PBRT infinite light, if the file lists one.
    std::optional<GltfEnvironment> environment;
};

// Appends a glTF file's meshes to the scene. Every (node, primitive) pair in
// the file's default scene becomes one Geom of type MESH whose transform is
// sceneTransform times the node's world transform; the primitive's vertices
// and triangles go into the scene's flat mesh arrays (see Scene). Its
// KHR_lights_punctual point and directional lights go into
// Scene::punctualLights under the same transforms, and so do a research
// scene's PBRT distant lights when info is given.
//
// Materials go into Scene::materials, unless materialOverride is a material
// index, which every primitive then uses: each glTF material becomes a
// Material with its factors and texture slots, the images the slots read go
// into Scene::textureImages and the slots themselves into Scene::textures.
//
// Only triangle-list primitives with float positions are loaded. Prints a
// summary line, or the error and returns false when the file cannot be
// loaded. info, when given, receives the rest of what the file holds; a file
// loaded without it (a mesh inside a scene JSON) has its PBRT infinite and
// distant lights reported as ignored.
bool loadGltf(const std::string& path, const glm::mat4& sceneTransform, int materialOverride, Scene& scene,
    GltfInfo* info = nullptr);
