#include "scene.h"

#include "gltf_loader.h"
#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include "json.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>

using namespace std;
using json = nlohmann::json;

namespace
{
// Render settings for a scene file that has none, the same as the scene
// JSONs in scenes/ use. The image is square unless the file's camera gives
// an aspect ratio.
const int DEFAULT_HEIGHT = 1024;
const int DEFAULT_ITERATIONS = 5000;
const int DEFAULT_TRACE_DEPTH = 8;
// Vertical field of view, in degrees, of the camera made for a glTF file
// that has none.
const float DEFAULT_FOVY = 45.0f;

// The command line wins over the scene file.
RenderSettings withOverrides(RenderSettings settings, const SceneOverrides& ov)
{
    if (ov.width > 0 && ov.height > 0)
    {
        settings.resolution = glm::ivec2(ov.width, ov.height);
    }
    if (ov.iterations > 0)
    {
        settings.iterations = ov.iterations;
    }
    if (ov.traceDepth > 0)
    {
        settings.traceDepth = ov.traceDepth;
    }
    return settings;
}

// A scene JSON array of three numbers. Throws, like json::at, when the key is
// missing or holds something else.
glm::vec3 vec3At(const json& object, const char* key)
{
    const json& v = object.at(key);
    if (!v.is_array() || v.size() != 3)
    {
        throw std::runtime_error(std::string(key) + " is not an array of three numbers");
    }
    return glm::vec3(v[0].get<float>(), v[1].get<float>(), v[2].get<float>());
}
}  // namespace

Scene::Scene(string filename, const SceneOverrides& ov)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    const string ext = lowercaseExtension(filename);
    if (ext == ".json")
    {
        // A missing key or a value of the wrong type throws while loading;
        // it ends the program here, with the scene file's name.
        try
        {
            loadFromJSON(filename, ov);
        }
        catch (const std::exception& e)
        {
            fatal("%s: %s", filename.c_str(), e.what());
        }
    }
    else if (ext == ".gltf" || ext == ".glb")
    {
        loadFromGltf(filename, ov);
    }
    else
    {
        fatal("%s: not a scene file (.json, .gltf or .glb)", filename.c_str());
    }
}

void Scene::bounds(size_t first, size_t last, glm::vec3& lo, glm::vec3& hi) const
{
    lo = glm::vec3(FLT_MAX);
    hi = glm::vec3(-FLT_MAX);
    auto grow = [&](const Geom& geom, const glm::vec3& p) {
        const glm::vec3 world = glm::vec3(geom.transform * glm::vec4(p, 1.0f));
        lo = glm::min(lo, world);
        hi = glm::max(hi, world);
    };
    for (size_t g = first; g < last; ++g)
    {
        const Geom& geom = geoms[g];
        if (geom.type == GeomType::MESH)
        {
            const TriangleMesh& mesh = meshes[geom.meshId];
            for (int t = 0; t < mesh.triCount; ++t)
            {
                const glm::ivec3 tri = indices[mesh.indexOffset + t];
                for (int k = 0; k < 3; ++k)
                {
                    grow(geom, positions[tri[k]]);
                }
            }
        }
        else
        {
            // A sphere has radius 0.5 and a cube half-width 0.5 before the transform.
            for (int c = 0; c < 8; ++c)
            {
                grow(geom, glm::vec3(c & 1 ? 0.5f : -0.5f, c & 2 ? 0.5f : -0.5f, c & 4 ? 0.5f : -0.5f));
            }
        }
    }
}

void Scene::loadFromJSON(const std::string& jsonName, const SceneOverrides& ov)
{
    std::ifstream f(jsonName);
    if (!f)
    {
        fatal("cannot open %s", jsonName.c_str());
    }
    const json data = json::parse(f);
    // Each material TYPE becomes parameters of the one material model
    // (Material), starting from glTF's default material:
    //   Diffuse     base color RGB and no specular layer: pure Lambert
    //   Emitting    emits RGB * EMITTANCE; a black base with no specular
    //               layer reflects nothing, so the path ends there
    //   Specular    metal with base color RGB and roughness ROUGHNESS
    //               (optional, 0 = a perfect mirror)
    //   Refractive  smooth glass of index IOR, the refraction tinted by RGB
    std::unordered_map<std::string, uint32_t> materialIdsByName;
    for (const auto& item : data.at("Materials").items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        const glm::vec3 rgb = vec3At(p, "RGB");
        const std::string type = p.at("TYPE");
        Material newMaterial{};
        if (type == "Diffuse")
        {
            newMaterial.baseColor = rgb;
            newMaterial.metallic = 0.0f;
            newMaterial.specularFactor = 0.0f;
        }
        else if (type == "Emitting")
        {
            newMaterial.baseColor = glm::vec3(0.0f);
            newMaterial.metallic = 0.0f;
            newMaterial.specularFactor = 0.0f;
            newMaterial.emission = rgb * p.at("EMITTANCE").get<float>();
        }
        else if (type == "Specular")
        {
            newMaterial.baseColor = rgb;
            newMaterial.metallic = 1.0f;
            newMaterial.roughness = p.value("ROUGHNESS", 0.0f);
        }
        else if (type == "Refractive")
        {
            newMaterial.baseColor = rgb;
            newMaterial.metallic = 0.0f;
            newMaterial.roughness = 0.0f;
            newMaterial.transmission = 1.0f;
            newMaterial.ior = p.at("IOR").get<float>();
        }
        else
        {
            fatal("%s: material %s has an unknown TYPE \"%s\" (Diffuse, Emitting, Specular or Refractive)",
                jsonName.c_str(), name.c_str(), type.c_str());
        }
        materialIdsByName[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    // Material by name. The map's operator[] would silently insert 0 for a
    // typo, which is the light in every Cornell scene.
    auto materialIndex = [&](const std::string& name) -> int {
        auto found = materialIdsByName.find(name);
        if (found == materialIdsByName.end())
        {
            fatal("%s: unknown material %s", jsonName.c_str(), name.c_str());
        }
        return (int)found->second;
    };

    // FILE paths in the scene are relative to the scene file's folder.
    const std::string sceneDir = jsonName.substr(0, jsonName.find_last_of("/\\") + 1);
    for (const auto& p : data.at("Objects"))
    {
        const std::string type = p.at("TYPE");
        const glm::mat4 transform =
            buildTransformationMatrix(vec3At(p, "TRANS"), vec3At(p, "ROTAT"), vec3At(p, "SCALE"));

        if (type == "mesh")
        {
            // A glTF file: its nodes and primitives become Geoms of type MESH
            // under this transform. MATERIAL, when given, replaces the file's
            // materials; without it they are appended to the material list.
            const int materialOverride = p.contains("MATERIAL") ? materialIndex(p.at("MATERIAL").get<std::string>()) : -1;
            const std::string file = sceneDir + p.at("FILE").get<std::string>();
            if (!loadGltf(file, transform, materialOverride, *this))
            {
                fatal("%s: cannot load %s", jsonName.c_str(), file.c_str());
            }
            continue;
        }

        Geom newGeom{};
        if (type == "cube")
        {
            newGeom.type = GeomType::CUBE;
        }
        else if (type == "sphere")
        {
            newGeom.type = GeomType::SPHERE;
        }
        else
        {
            fatal("%s: unknown object TYPE \"%s\" (cube, sphere or mesh)", jsonName.c_str(), type.c_str());
        }
        newGeom.materialId = materialIndex(p.at("MATERIAL").get<std::string>());
        newGeom.meshId = -1;
        newGeom.transform = transform;
        newGeom.inverseTransform = glm::inverse(transform);
        newGeom.invTranspose = glm::inverseTranspose(transform);

        geoms.push_back(newGeom);
    }
    const json& cameraData = data.at("Camera");
    const json& res = cameraData.at("RES");
    RenderSettings settings;
    settings.resolution = glm::ivec2(res.at(0).get<int>(), res.at(1).get<int>());
    settings.iterations = cameraData.at("ITERATIONS").get<int>();
    settings.traceDepth = cameraData.at("DEPTH").get<int>();
    settings.imageName = cameraData.at("FILE").get<std::string>();

    CameraPose pose;
    pose.eye = vec3At(cameraData, "EYE");
    pose.lookAt = vec3At(cameraData, "LOOKAT");
    pose.up = vec3At(cameraData, "UP");
    pose.fovy = cameraData.at("FOVY").get<float>();
    pose.mirrored = false;

    initRenderState(withOverrides(settings, ov), pose);
}

void Scene::loadFromGltf(const std::string& gltfName, const SceneOverrides& ov)
{
    GltfInfo info;
    if (!loadGltf(gltfName, glm::mat4(1.0f), -1, *this, &info))
    {
        fatal("cannot load %s", gltfName.c_str());
    }
    if (geoms.empty())
    {
        fatal("%s: nothing to render", gltfName.c_str());
    }
    // Only a surface that emits lights a scene so far.
    if (std::none_of(materials.begin(), materials.end(), [](const Material& m) {
            return maxComponent(m.emission) > 0.0f;
        }))
    {
        cerr << "No emissive material in " << gltfName << ": the render will be black" << endl;
    }

    // The file's hints where it has them, the defaults elsewhere.
    const GltfRenderHints& hints = info.render;
    RenderSettings settings;
    settings.resolution = glm::ivec2(DEFAULT_HEIGHT);
    if (hints.width > 0 && hints.height > 0)
    {
        settings.resolution = glm::ivec2(hints.width, hints.height);
    }
    else if (info.camera && info.camera->aspectRatio > 0.0f)
    {
        settings.resolution.x = glm::max(1, (int)std::round(DEFAULT_HEIGHT * info.camera->aspectRatio));
    }
    settings.iterations = DEFAULT_ITERATIONS;
    // PBRT's max_depth counts bounces, and its paths sample the lights at
    // every bounce. A path here has to hit a light, which takes one more ray
    // after the last bounce.
    settings.traceDepth = hints.maxDepth > 0 ? hints.maxDepth + 1 : DEFAULT_TRACE_DEPTH;
    settings.imageName = std::filesystem::path(gltfName).stem().string();
    settings = withOverrides(settings, ov);

    // The scene's bounding sphere, from the box's center and half diagonal.
    const glm::vec3 center = 0.5f * (info.boundsMin + info.boundsMax);
    const float radius = glm::max(0.5f * glm::length(info.boundsMax - info.boundsMin), 1e-3f);

    CameraPose pose;
    if (info.camera)
    {
        const GltfCamera& c = *info.camera;
        pose.eye = c.position;
        pose.up = c.up;
        pose.fovy = glm::degrees(c.yfov);
        pose.mirrored = c.mirrored;

        // A glTF camera has no look-at point and the interactive camera needs
        // one to orbit: the point on the view axis nearest the middle of the
        // scene, or one scene radius ahead when the middle is beside or
        // behind the camera.
        float ahead = glm::dot(center - c.position, c.view);
        if (ahead < 0.01f * radius)
        {
            ahead = radius;
        }
        pose.lookAt = c.position + ahead * c.view;

        // The viewport camera (applyPose in main.cpp) keeps world +Y up, so
        // its right is always level: perpendicular to view and +Y, or +X
        // when it looks straight up or down. A glTF camera whose right
        // points elsewhere is rolled about its view axis, and the roll is
        // lost.
        glm::vec3 level = glm::cross(c.view, glm::vec3(0.0f, 1.0f, 0.0f));
        level = glm::length(level) < 1e-3f ? glm::vec3(1.0f, 0.0f, 0.0f) : glm::normalize(level);
        if (glm::dot(glm::normalize(glm::cross(c.view, c.up)), level) < 0.9999f)
        {
            cerr << "The camera in " << gltfName << " is rolled about its view axis; the roll is not kept" << endl;
        }
    }
    else
    {
        // No camera in the file: look at the whole scene from +Z, far enough
        // back that the bounding sphere fits the narrower side of the image.
        const float halfY = glm::radians(0.5f * DEFAULT_FOVY);
        const float aspect = (float)settings.resolution.x / (float)settings.resolution.y;
        const float halfAngle = glm::min(halfY, std::atan(std::tan(halfY) * aspect));
        pose.eye = center + glm::vec3(0.0f, 0.0f, radius / std::sin(halfAngle));
        pose.lookAt = center;
        pose.up = glm::vec3(0.0f, 1.0f, 0.0f);
        pose.fovy = DEFAULT_FOVY;
        pose.mirrored = false;
    }

    initRenderState(settings, pose);
}

void Scene::initRenderState(const RenderSettings& settings, const CameraPose& pose)
{
    Camera& camera = state.camera;
    camera.resolution = settings.resolution;
    state.iterations = settings.iterations;
    state.traceDepth = settings.traceDepth;
    state.imageName = settings.imageName;

    camera.position = pose.eye;
    camera.lookAt = pose.lookAt;
    camera.up = pose.up;
    camera.mirrored = pose.mirrored;

    // fovy is the full vertical field of view in degrees, so the half angle
    // sets the image plane's half height at unit distance.
    float yscaled = tan(0.5f * pose.fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    // Orthonormal basis from eye, lookAt and up: view first, right from
    // view and up, then up rebuilt so it is perpendicular to both.
    camera.view = glm::normalize(camera.lookAt - camera.position);
    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.up = glm::normalize(glm::cross(camera.right, camera.view));
    if (camera.mirrored)
    {
        camera.right = -camera.right;
    }

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}
