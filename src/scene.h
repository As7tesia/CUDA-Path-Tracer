#pragma once

#include "sceneStructs.h"
#include <string>
#include <vector>

// Command-line overrides for values normally read from the scene file.
// 0 means "use the scene file's value".
struct SceneOverrides
{
    int width = 0;
    int height = 0;
    int iterations = 0;
    int traceDepth = 0;
};

// What a scene file says about the render itself. A scene JSON states all of
// it in its Camera block; glTF has no such settings, so a glTF scene starts
// from the defaults in scene.cpp.
struct RenderSettings
{
    glm::ivec2 resolution;
    int iterations;
    int traceDepth;
    std::string imageName;  // base name of auto-saved images
};

// Where the camera is and what it sees.
struct CameraPose
{
    glm::vec3 eye;
    glm::vec3 lookAt;  // also the point the interactive camera orbits
    glm::vec3 up;
    float fovy;        // full vertical field of view, degrees
    bool mirrored;     // see Camera::mirrored
};

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName, const SceneOverrides& ov);
    void loadFromGltf(const std::string& gltfName, const SceneOverrides& ov);
    void initRenderState(const RenderSettings& settings, const CameraPose& pose);
public:
    // filename is a scene JSON, or a glTF file (.gltf / .glb) that is the
    // whole scene: geometry, materials, lights and camera.
    Scene(std::string filename, const SceneOverrides& ov = SceneOverrides());

    // World-space bounding box of geoms [first, last): the transformed
    // corners of the unit cube around a sphere or cube, the transformed
    // vertices of a mesh. lo > hi when the range is empty.
    void bounds(size_t first, size_t last, glm::vec3& lo, glm::vec3& hi) const;

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    // Triangle meshes from glTF files, flattened: every primitive appends its
    // vertices and triangles here and gets a TriangleMesh saying where its
    // triangles are. Empty when the scene has no meshes.
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> uvs;
    std::vector<glm::vec4> tangents;
    std::vector<glm::ivec3> indices;
    std::vector<TriangleMesh> meshes;
    // Images and the textures that read them, from glTF materials. Empty
    // when no material has a texture.
    std::vector<TextureImage> textureImages;
    std::vector<Texture> textures;
    RenderState state;
};
