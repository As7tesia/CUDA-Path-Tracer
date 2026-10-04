#pragma once

#include "scene/environment.h"
#include "scene/sceneStructs.h"
#include <string>
#include <vector>

// Command-line overrides for values normally read from the scene file.
// 0 (or an empty string) means "use the scene file's value".
struct SceneOverrides
{
    int width = 0;
    int height = 0;
    int iterations = 0;
    int traceDepth = 0;
    // A lat-long .hdr or .exr that replaces the scene's environment, at
    // strength 1 and no rotation
    std::string environmentFile;
    // MIS compensation in the environment's sampling table (environment.h);
    // --no-env-compensation turns it off
    bool environmentCompensation = true;
    // Whether the environment joins the light list; --no-env-nee leaves it
    // to BSDF sampling, with next event estimation for the other lights
    bool environmentLight = true;
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

// The scene file a command-line argument names: the argument itself when it
// is an existing file; for a bare name (no folder, no extension)
// scenes/<name>.json, then the name's entry in scenes/catalog.json, which
// gives short names to the glTF scenes under scenes/assets. Paths are
// relative to the working directory, the repository root. Throws a
// std::runtime_error saying what was tried when nothing matches.
std::string findSceneFile(const std::string& argument);

// Prints the names findSceneFile takes: the scene JSONs in scenes/ and the
// catalog's entries, marking those whose file is not downloaded.
void listScenes();

// A name findSceneFile takes, for the window's scene list. available is
// false for a catalog entry whose glTF is not downloaded.
struct SceneName
{
    std::string name;
    bool available;
};
// The scene JSONs in scenes/, sorted, then the catalog's entries in file
// order. Empty when there is no scenes/ folder.
std::vector<SceneName> sceneNames();

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName, const SceneOverrides& ov);
    void loadFromGltf(const std::string& gltfName, const SceneOverrides& ov);
    void initRenderState(const RenderSettings& settings, const CameraPose& pose);
public:
    // Fills the light list below once the scene and its environment are
    // loaded (lights.cpp). Called again when the window swaps the
    // environment, since its power changes every light's share.
    void buildLights();
    // filename is a scene JSON, or a glTF file (.gltf / .glb) that is the
    // whole scene: geometry, materials, lights and camera. Throws a
    // std::runtime_error, starting with the file's name, when the file
    // cannot be read or is not a scene.
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
    // The texture a base color slot reads when its image is missing or
    // cannot be decoded: one magenta texel, made by the glTF loader on first
    // use (-1 until then) and shared by every glTF file of the scene.
    int missingTexture = -1;
    // The scene file's own light from outside the scene: a scene JSON's
    // Environment block, or a glTF research scene's PBRT infinite light.
    EnvironmentSource environmentSource;
    // The environment that renders, read: --env when given, else the scene
    // file's own. The window can replace it.
    Environment environment;

    // Next event estimation's lights. punctualLights comes from the glTF
    // loader (KHR_lights_punctual point and directional lights, PBRT distant
    // lights); buildLights adds every emissive triangle (meshes, and cubes as
    // their 12 triangles; emissive spheres are left to BSDF sampling) and the
    // probabilities the shade kernel picks lights with, proportional to
    // their power. Empty when the scene has no light to sample.
    std::vector<PunctualLight> punctualLights;
    std::vector<LightTriangle> lightTriangles;
    // Running sum of the pick probabilities over lightTriangles, then
    // punctualLights, then the environment when it is one of the lights;
    // the last entry is 1.
    std::vector<float> lightCdf;
    // The environment's pick probability, the last entry's share of
    // lightCdf; 0 when the scene has no environment to sample, or
    // environmentLight is off.
    float environmentPickPdf = 0.0f;
    bool environmentLight = true;  // SceneOverrides::environmentLight
    // Per material: the density per unit area of next event estimation
    // landing on a point of one of its triangles, the pick probability over
    // the area, which comes to the same value for every triangle of the
    // material (lights.cpp). The light pdf of a ray that hits an emitter
    // needs nothing else. 0 for a material NEE never samples.
    std::vector<float> emitterAreaPdf;
    RenderState state;
};
