// The light list next event estimation picks from (see Scene in scene.h).
//
// Every light gets a share of the picks proportional to its power, the way
// PBRT's power light sampler weighs them, so the brighter and larger lights
// get more of the samples:
//
//   emissive triangle   2 pi * area * luminance(emission), emitting from both sides
//   point light         4 pi * luminance(intensity)
//   distant light       pi * r^2 * luminance(irradiance), r the scene's bounding sphere
//
// A triangle's power uses its material's emission factor and ignores the
// emissive texture, so a point is picked uniformly over the triangle's area.
// The area then cancels out of the density per unit area,
//
//   P_pick / area = 2 pi * luminance(emission) / total power,
//
// which is one number per material (Scene::emitterAreaPdf).

#include "scene/scene.h"

#include "utilities.h"

#include <cstdio>
#include <unordered_map>
#include <utility>

namespace
{
// The unit cube's corners (+-0.5) and its 12 triangles, wound
// counterclockwise seen from outside, the same as the OptiX cube GAS.
glm::vec3 cubeCorner(int i)
{
    return glm::vec3((i & 1) ? 0.5f : -0.5f, (i & 2) ? 0.5f : -0.5f, (i & 4) ? 0.5f : -0.5f);
}

const int CUBE_TRIANGLES[12][3] = {
    {0, 4, 6}, {0, 6, 2},   // -x
    {1, 7, 5}, {1, 3, 7},   // +x
    {0, 5, 4}, {0, 1, 5},   // -y
    {2, 6, 7}, {2, 7, 3},   // +y
    {0, 2, 3}, {0, 3, 1},   // -z
    {4, 7, 6}, {4, 5, 7},   // +z
};
}  // namespace

void Scene::buildLights()
{
    // An emissive sphere reaches the image through BSDF sampling only, so
    // its material has to tell the shade kernel that NEE never samples it.
    // A sphere that shares its material with a cube or a mesh gets its own
    // copy of it.
    std::vector<bool> onTriangles(materials.size(), false);
    for (const Geom& g : geoms)
    {
        if (g.type != GeomType::SPHERE)
        {
            onTriangles[g.materialId] = true;
        }
    }
    std::unordered_map<int, int> sphereCopies;
    for (Geom& g : geoms)
    {
        if (g.type != GeomType::SPHERE || maxComponent(materials[g.materialId].emission) <= 0.0f
            || !onTriangles[g.materialId])
        {
            continue;
        }
        auto found = sphereCopies.find(g.materialId);
        if (found == sphereCopies.end())
        {
            const Material copy = materials[g.materialId];  // a copy first: push_back may reallocate
            materials.push_back(copy);
            found = sphereCopies.emplace(g.materialId, (int)materials.size() - 1).first;
        }
        g.materialId = found->second;
    }

    // Every triangle of an emissive cube or mesh, in world space.
    lightTriangles.clear();
    std::vector<double> power;
    auto addTriangle = [&](const Geom& g, glm::vec3 p0, glm::vec3 p1, glm::vec3 p2, glm::vec2 uv0, glm::vec2 uv1,
                           glm::vec2 uv2, bool mirrored) {
        p0 = glm::vec3(g.transform * glm::vec4(p0, 1.0f));
        p1 = glm::vec3(g.transform * glm::vec4(p1, 1.0f));
        p2 = glm::vec3(g.transform * glm::vec4(p2, 1.0f));
        // A mirroring transform turns the winding around; swapping two
        // corners turns it back, so cross(e1, e2) stays the front face.
        if (mirrored)
        {
            std::swap(p1, p2);
            std::swap(uv1, uv2);
        }
        LightTriangle t;
        t.p0 = p0;
        t.e1 = p1 - p0;
        t.e2 = p2 - p0;
        t.uv0 = uv0;
        t.uv1 = uv1;
        t.uv2 = uv2;
        t.materialId = g.materialId;
        const double area = 0.5 * (double)glm::length(glm::cross(t.e1, t.e2));
        if (area > 0.0)  // a degenerate triangle has nothing to pick
        {
            lightTriangles.push_back(t);
            power.push_back(2.0 * PI * area * luminance(materials[g.materialId].emission));
        }
    };
    for (const Geom& g : geoms)
    {
        if (g.type == GeomType::SPHERE || maxComponent(materials[g.materialId].emission) <= 0.0f)
        {
            continue;
        }
        const bool mirrored = glm::determinant(glm::mat3(g.transform)) < 0.0f;
        if (g.type == GeomType::CUBE)
        {
            for (const auto& tri : CUBE_TRIANGLES)
            {
                addTriangle(g, cubeCorner(tri[0]), cubeCorner(tri[1]), cubeCorner(tri[2]), glm::vec2(0.0f),
                    glm::vec2(0.0f), glm::vec2(0.0f), mirrored);
            }
        }
        else
        {
            const TriangleMesh& mesh = meshes[g.meshId];
            for (int i = 0; i < mesh.triCount; ++i)
            {
                const glm::ivec3 tri = indices[mesh.indexOffset + i];
                addTriangle(g, positions[tri.x], positions[tri.y], positions[tri.z], uvs[tri.x], uvs[tri.y],
                    uvs[tri.z], mirrored);
            }
        }
    }

    // Punctual lights. A distant light's power is what it pours onto a disk
    // as large as the scene.
    glm::vec3 lo;
    glm::vec3 hi;
    bounds(0, geoms.size(), lo, hi);
    const double radius = lo.x <= hi.x ? 0.5 * (double)glm::length(hi - lo) : 1.0;
    int pointLights = 0;
    for (const PunctualLight& light : punctualLights)
    {
        if (light.type == LIGHT_POINT)
        {
            ++pointLights;
            power.push_back(4.0 * PI * luminance(light.intensity));
        }
        else
        {
            power.push_back(PI * radius * radius * luminance(light.intensity));
        }
    }

    double total = 0.0;
    for (double p : power)
    {
        total += p;
    }
    lightCdf.clear();
    emitterAreaPdf.assign(materials.size(), 0.0f);
    if (total <= 0.0)
    {
        lightTriangles.clear();
        punctualLights.clear();
        return;
    }

    // Summed in double: in a scene with millions of emissive triangles each
    // one's share is near float's resolution next to 1.
    lightCdf.resize(power.size());
    double running = 0.0;
    for (size_t i = 0; i < power.size(); ++i)
    {
        running += power[i];
        lightCdf[i] = (float)(running / total);
    }
    lightCdf.back() = 1.0f;

    for (const LightTriangle& t : lightTriangles)
    {
        emitterAreaPdf[t.materialId] = (float)(2.0 * PI * luminance(materials[t.materialId].emission) / total);
    }
    for (size_t i = 0; i < punctualLights.size(); ++i)
    {
        punctualLights[i].pickPdf = (float)(power[lightTriangles.size() + i] / total);
    }

    printf("Lights for next event estimation: %zu emissive triangles, %d point, %zu distant\n", lightTriangles.size(),
        pointLights, punctualLights.size() - pointLights);
}
