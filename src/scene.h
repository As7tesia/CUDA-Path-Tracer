#pragma once

#include "sceneStructs.h"
#include <vector>

// Command-line overrides for values normally read from the scene JSON.
// 0 / empty means "use the scene file's value".
struct SceneOverrides
{
    int width = 0;
    int height = 0;
    int iterations = 0;
};

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName, const SceneOverrides& ov);
public:
    Scene(std::string filename, const SceneOverrides& ov = SceneOverrides());

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    // Triangle meshes from glTF files, flattened: every primitive appends its
    // vertices and triangles here and gets a TriangleMesh saying where its
    // triangles are. Empty when the scene has no meshes.
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::ivec3> indices;
    std::vector<TriangleMesh> meshes;
    RenderState state;
};
