#pragma once

#include <glm/glm.hpp>

#include <string>

class Scene;

// Appends a glTF file's meshes to the scene. Every (node, primitive) pair in
// the file's default scene becomes one Geom of type MESH whose transform is
// sceneTransform times the node's world transform; the primitive's vertices
// and triangles go into the scene's flat mesh arrays (see Scene) and its
// material into Scene::materials as a Diffuse with the base color factor,
// unless materialOverride is a material index, which every primitive then
// uses. Only triangle-list primitives with float positions are loaded;
// textures are not read yet. Prints a summary line, or the error and returns
// false when the file cannot be loaded.
bool loadGltf(const std::string& path, const glm::mat4& sceneTransform, int materialOverride, Scene& scene);
