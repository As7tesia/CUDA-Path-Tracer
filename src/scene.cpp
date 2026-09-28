#include "scene.h"

#include "gltf_loader.h"
#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

using namespace std;
using json = nlohmann::json;

Scene::Scene(string filename, const SceneOverrides& ov)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename, ov);
        return;
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
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];

    // CLI overrides must land before the fov / pixelLength math below
    if (ov.width > 0 && ov.height > 0)
    {
        camera.resolution = glm::ivec2(ov.width, ov.height);
    }
    if (ov.iterations > 0)
    {
        state.iterations = ov.iterations;
    }
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    // FOVY is the full vertical field of view in degrees, so the half angle
    // sets the image plane's half height at unit distance.
    float yscaled = tan(0.5f * fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (2 * atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    // Orthonormal basis from EYE, LOOKAT and UP: view first, right from
    // view and UP, then up rebuilt so it is perpendicular to both.
    camera.view = glm::normalize(camera.lookAt - camera.position);
    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.up = glm::normalize(glm::cross(camera.right, camera.view));

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}
