#include "scene.h"

#include "gltf_loader.h"
#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
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
}  // namespace

Scene::Scene(string filename, const SceneOverrides& ov)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    string ext = std::filesystem::path(filename).extension().string();
    for (char& c : ext)
    {
        c = (char)tolower((unsigned char)c);
    }
    if (ext == ".json")
    {
        loadFromJSON(filename, ov);
    }
    else if (ext == ".gltf" || ext == ".glb")
    {
        loadFromGltf(filename, ov);
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

void Scene::loadFromJSON(const std::string& jsonName, const SceneOverrides& ov)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        // TODO: handle materials loading differently
        const auto& col = p["RGB"];
        newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        if (p["TYPE"] == "Diffuse")
        {
            newMaterial.type = DIFFUSE;
        }
        else if (p["TYPE"] == "Emitting")
        {
            newMaterial.type = EMISSIVE;
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (p["TYPE"] == "Specular")
        {
            newMaterial.type = SPECULAR;
            newMaterial.specular.color = newMaterial.color;
        }
        else if (p["TYPE"] == "Refractive")
        {
            newMaterial.type = REFRACTIVE;
            newMaterial.ior = p["IOR"];
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    // Material by name. The map's operator[] would silently insert 0 for a
    // typo, which is the light in every Cornell scene.
    auto materialIndex = [&](const std::string& name) -> int {
        auto found = MatNameToID.find(name);
        if (found == MatNameToID.end())
        {
            cout << "Unknown material " << name << endl;
            exit(-1);
        }
        return (int)found->second;
    };

    // FILE paths in the scene are relative to the scene file's folder.
    const std::string sceneDir = jsonName.substr(0, jsonName.find_last_of("/\\") + 1);
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        const glm::vec3 translation(trans[0], trans[1], trans[2]);
        const glm::vec3 rotation(rotat[0], rotat[1], rotat[2]);
        const glm::vec3 scaling(scale[0], scale[1], scale[2]);
        const glm::mat4 transform = utilityCore::buildTransformationMatrix(translation, rotation, scaling);

        if (type == "mesh")
        {
            // A glTF file: its nodes and primitives become Geoms of type MESH
            // under this transform. MATERIAL, when given, replaces the file's
            // materials; without it they are appended to the material list.
            if (!p.contains("FILE"))
            {
                cout << "Mesh object without a FILE" << endl;
                exit(-1);
            }
            const int materialOverride = p.contains("MATERIAL") ? materialIndex(p["MATERIAL"].get<std::string>()) : -1;
            if (!loadGltf(sceneDir + std::string(p["FILE"]), transform, materialOverride, *this))
            {
                exit(-1);
            }
            continue;
        }

        Geom newGeom{};
        newGeom.type = (type == "cube") ? CUBE : SPHERE;
        newGeom.materialid = materialIndex(p["MATERIAL"].get<std::string>());
        newGeom.meshId = -1;
        newGeom.translation = translation;
        newGeom.rotation = rotation;
        newGeom.scale = scaling;
        newGeom.transform = transform;
        newGeom.inverseTransform = glm::inverse(transform);
        newGeom.invTranspose = glm::inverseTranspose(transform);

        geoms.push_back(newGeom);
    }
    const auto& cameraData = data["Camera"];
    const auto& res = cameraData["RES"];
    RenderSettings settings;
    settings.resolution = glm::ivec2(res[0].get<int>(), res[1].get<int>());
    settings.iterations = cameraData["ITERATIONS"].get<int>();
    settings.traceDepth = cameraData["DEPTH"].get<int>();
    settings.imageName = cameraData["FILE"].get<std::string>();

    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    CameraPose pose;
    pose.eye = glm::vec3(pos[0], pos[1], pos[2]);
    pose.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    pose.up = glm::vec3(up[0], up[1], up[2]);
    pose.fovy = cameraData["FOVY"];
    pose.mirrored = false;

    initRenderState(withOverrides(settings, ov), pose);
}

void Scene::loadFromGltf(const std::string& gltfName, const SceneOverrides& ov)
{
    GltfInfo info;
    if (!loadGltf(gltfName, glm::mat4(1.0f), -1, *this, &info))
    {
        exit(-1);
    }
    if (geoms.empty())
    {
        cout << "Nothing to render in " << gltfName << endl;
        exit(-1);
    }
    // Only a surface that emits lights a scene so far.
    if (std::none_of(materials.begin(), materials.end(), [](const Material& m) { return m.emittance > 0.0f; }))
    {
        cout << "No emissive material in " << gltfName << ": the render will be black" << endl;
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

        // The interactive camera (updateCameraFromOrbit in main.cpp) orbits
        // with world +Y up, so its right is always level: perpendicular to
        // view and +Y, or +X when it looks straight up or down. A glTF camera
        // whose right points elsewhere is rolled about its view axis, and
        // the roll is lost.
        glm::vec3 level = glm::cross(c.view, glm::vec3(0.0f, 1.0f, 0.0f));
        level = glm::length(level) < 1e-3f ? glm::vec3(1.0f, 0.0f, 0.0f) : glm::normalize(level);
        if (glm::dot(glm::normalize(glm::cross(c.view, c.up)), level) < 0.9999f)
        {
            cout << "The camera in " << gltfName << " is rolled about its view axis; the roll is not kept" << endl;
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
    float fovx = (2 * atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, pose.fovy);
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
