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
    RenderState state;
};
