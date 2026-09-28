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
};

// Appends a glTF file's meshes to the scene. Every (node, primitive) pair in
// the file's default scene becomes one Geom of type MESH whose transform is
// sceneTransform times the node's world transform; the primitive's vertices
// and triangles go into the scene's flat mesh arrays (see Scene).
//
// Materials go into Scene::materials, unless materialOverride is a material
// index, which every primitive then uses. A material that emits (emissive
// factor times KHR_materials_emissive_strength) becomes an Emitting material,
// any other a Diffuse with the base color factor.
//
// Only triangle-list primitives with float positions are loaded; textures
// are not read yet. Prints a summary line, or the error and returns false
// when the file cannot be loaded. info, when given, receives the rest of
// what the file holds.
bool loadGltf(const std::string& path, const glm::mat4& sceneTransform, int materialOverride, Scene& scene,
    GltfInfo* info = nullptr);
