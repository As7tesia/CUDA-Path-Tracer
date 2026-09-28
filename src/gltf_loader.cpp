// glTF loading through tinygltf. See gltf_loader.h.
//
// This is the one translation unit that compiles tinygltf's implementation.
// The repo already has nlohmann's json.hpp, so tinygltf uses that instead of
// its own copy. Textures are not loaded yet, so image decoding is compiled
// out: image files referenced by URI are skipped (only the URI is kept),
// embedded images go through the no-op callback below, and the glTF writer
// is off. This also sidesteps the repo's stb_image being older than the
// 16-bit loaders tinygltf's built-in image path calls.

#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_INCLUDE_JSON
#define TINYGLTF_NO_STB_IMAGE
#define TINYGLTF_NO_STB_IMAGE_WRITE
#define TINYGLTF_NO_EXTERNAL_IMAGE
#include "json.hpp"
#include "tiny_gltf.h"

#include "gltf_loader.h"
#include "scene.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include <cctype>
#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace
{
// Image callback for images embedded in the file (data URIs, GLB chunks):
// keeps the image entry and decodes nothing. External image files never get
// here because of TINYGLTF_NO_EXTERNAL_IMAGE.
bool skipImage(tinygltf::Image*, const int, std::string*, std::string*, int, int, const unsigned char*, int, void*)
{
    return true;
}

// A window onto one accessor's elements: glTF stores vertex attributes in
// buffer views that may interleave several attributes, so element i sits at
// data + i * stride, not at i * sizeof(element).
struct AccessorView
{
    const unsigned char* data = nullptr;
    size_t stride = 0;
    size_t count = 0;
    int componentType = 0;   // TINYGLTF_COMPONENT_TYPE_*
    int numComponents = 0;   // 1 for SCALAR, 3 for VEC3, ...

    const unsigned char* at(size_t i) const { return data + i * stride; }
};

bool viewAccessor(const tinygltf::Model& model, int accessorIndex, AccessorView& view, std::string& err)
{
    if (accessorIndex < 0 || accessorIndex >= (int)model.accessors.size())
    {
        err = "accessor index out of range";
        return false;
    }
    const tinygltf::Accessor& a = model.accessors[accessorIndex];
    if (a.bufferView < 0 || a.bufferView >= (int)model.bufferViews.size())
    {
        err = "accessor without a buffer view (sparse, Draco or meshopt data is not supported)";
        return false;
    }
    if (a.sparse.isSparse)
    {
        err = "sparse accessors are not supported";
        return false;
    }
    const tinygltf::BufferView& bv = model.bufferViews[a.bufferView];
    // tinygltf checks this index for images only, not for mesh data.
    if (bv.buffer < 0 || bv.buffer >= (int)model.buffers.size())
    {
        err = "buffer view with a bad buffer index";
        return false;
    }
    const tinygltf::Buffer& buffer = model.buffers[bv.buffer];
    const int stride = a.ByteStride(bv);  // the view's byteStride, or the packed element size
    const int componentSize = tinygltf::GetComponentSizeInBytes(a.componentType);
    const int numComponents = tinygltf::GetNumComponentsInType(a.type);
    if (stride <= 0 || componentSize <= 0 || numComponents <= 0)
    {
        err = "accessor with an unsupported layout";
        return false;
    }
    // count is bounded by the buffer size first so the product below cannot wrap.
    const size_t begin = bv.byteOffset + a.byteOffset;
    const size_t end = a.count == 0 ? begin : begin + (a.count - 1) * stride + componentSize * numComponents;
    if (a.count > buffer.data.size() || end > buffer.data.size())
    {
        err = "accessor reaches past the end of its buffer";
        return false;
    }
    view.data = buffer.data.data() + begin;
    view.stride = stride;
    view.count = a.count;
    view.componentType = a.componentType;
    view.numComponents = numComponents;
    return true;
}

bool isFloatVec3(const AccessorView& v)
{
    return v.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT && v.numComponents == 3;
}

glm::vec3 readVec3(const AccessorView& v, size_t i)
{
    const float* f = reinterpret_cast<const float*>(v.at(i));
    return glm::vec3(f[0], f[1], f[2]);
}

// Index accessors come in three widths, all unsigned; readIndex assumes one of them.
bool isIndexType(int componentType)
{
    return componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE
        || componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT
        || componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT;
}

unsigned int readIndex(const AccessorView& v, size_t i)
{
    const unsigned char* p = v.at(i);
    switch (v.componentType)
    {
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
        return *p;
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
        return *reinterpret_cast<const uint16_t*>(p);
    default:
        return *reinterpret_cast<const uint32_t*>(p);
    }
}

// A node's transform relative to its parent: either a full matrix or
// translation * rotation * scale, each part optional.
glm::mat4 nodeLocalMatrix(const tinygltf::Node& node)
{
    if (node.matrix.size() == 16)
    {
        // glTF stores matrices column-major, the same as glm.
        glm::mat4 m;
        for (int i = 0; i < 16; ++i)
        {
            m[i / 4][i % 4] = (float)node.matrix[i];
        }
        return m;
    }
    glm::mat4 t(1.0f);
    glm::mat4 r(1.0f);
    glm::mat4 s(1.0f);
    if (node.translation.size() == 3)
    {
        t = glm::translate(glm::mat4(1.0f),
            glm::vec3((float)node.translation[0], (float)node.translation[1], (float)node.translation[2]));
    }
    if (node.rotation.size() == 4)
    {
        // glTF quaternions are (x, y, z, w); glm's constructor takes w first.
        r = glm::mat4_cast(glm::quat((float)node.rotation[3],
            (float)node.rotation[0], (float)node.rotation[1], (float)node.rotation[2]));
    }
    if (node.scale.size() == 3)
    {
        s = glm::scale(glm::mat4(1.0f),
            glm::vec3((float)node.scale[0], (float)node.scale[1], (float)node.scale[2]));
    }
    return t * r * s;
}

// Extensions and extras reach this file as raw JSON values. object[key], or
// a null value when object is not a JSON object or has no such member
// (Value::Get asserts on anything that is not an object).
const tinygltf::Value& member(const tinygltf::Value& object, const char* key)
{
    static const tinygltf::Value null;
    return object.Has(key) ? object.Get(key) : null;
}

float numberMember(const tinygltf::Value& object, const char* key, float fallback)
{
    const tinygltf::Value& value = member(object, key);
    return value.IsNumber() ? (float)value.GetNumberAsDouble() : fallback;
}

// A number from one of a material's extensions, or fallback when the
// material does not have the extension or the extension does not say.
float extensionNumber(const tinygltf::Material& material, const char* extension, const char* key, float fallback)
{
    auto found = material.extensions.find(extension);
    return found == material.extensions.end() ? fallback : numberMember(found->second, key, fallback);
}

// The render settings of a research scene, see GltfRenderHints. Two are left
// alone. fov_degrees is PBRT's angle across the shorter side of the image,
// and the camera's yfov already is the vertical angle. samples_per_pixel is
// tuned for PBRT, whose paths sample the lights and need far fewer samples.
GltfRenderHints renderHints(const tinygltf::Model& model)
{
    GltfRenderHints hints;
    const tinygltf::Value& render = member(member(model.extras, "pbrt"), "render");
    const tinygltf::Value& resolution = member(render, "resolution");
    if (resolution.ArrayLen() == 2)
    {
        hints.width = resolution.Get(0).GetNumberAsInt();
        hints.height = resolution.Get(1).GetNumberAsInt();
    }
    hints.maxDepth = (int)numberMember(render, "max_depth", 0.0f);
    return hints;
}

// Names the lights in the file that do not reach the render, so a dark image
// explains itself. Only emissive surfaces light a scene so far: punctual
// lights (KHR_lights_punctual, which tinygltf parses into model.lights) and
// PBRT's distant lights have no surface a path could hit and wait for direct
// light sampling, PBRT's infinite lights wait for the environment map. The
// research scenes list the PBRT ones under extras.pbrt.light_sources and
// mark those glTF cannot express as "metadata_only"; the list's area lights
// are in the file as emissive meshes.
void reportUnusedLights(const tinygltf::Model& model, const std::string& path)
{
    if (!model.lights.empty())
    {
        fprintf(stderr, "glTF %s: %zu punctual lights ignored (not supported yet)\n", path.c_str(), model.lights.size());
    }
    const tinygltf::Value& sources = member(member(model.extras, "pbrt"), "light_sources");
    for (size_t i = 0; i < sources.ArrayLen(); ++i)
    {
        const tinygltf::Value& kind = member(sources.Get(i), "pbrt");
        const tinygltf::Value& form = member(sources.Get(i), "gltf");
        if (form.IsString() && form.Get<std::string>() == "metadata_only")
        {
            fprintf(stderr, "glTF %s: PBRT %s light ignored (not supported yet)\n", path.c_str(),
                kind.IsString() ? kind.Get<std::string>().c_str() : "unnamed");
        }
    }
}

// State for one file: the model, the scene being filled, and the maps that
// keep a primitive or a material shared by several nodes as one entry.
struct Loader
{
    const tinygltf::Model& model;
    const std::string& path;
    Scene& scene;
    int materialOverride;
    std::map<int, int> materialIds;                    // glTF material index (-1 = none) -> Scene material
    std::map<std::pair<int, int>, int> meshIds;        // (glTF mesh, primitive) -> Scene mesh, -1 if unusable
    std::optional<GltfCamera> camera;                  // the first usable camera the walk reached

    // A primitive this loader cannot use is skipped, not fatal: the rest of
    // the file still loads.
    void warn(const char* what, int meshIndex, int primIndex)
    {
        fprintf(stderr, "glTF %s: mesh %d primitive %d skipped: %s\n", path.c_str(), meshIndex, primIndex, what);
    }

    // glTF materials become Diffuse with the base color factor for now; the
    // base color texture and the metallic-roughness inputs wait for the GGX
    // step. A primitive without a material gets glTF's default, white.
    //
    // A material that emits becomes Emitting. glTF's emission is the emissive
    // factor times the emissive texture times the strength from
    // KHR_materials_emissive_strength (1 without the extension), which maps
    // onto color times emittance. Two things are missing until the material
    // model grows in the GGX step:
    // - An emitter only emits. The shade kernel ends a path at a light, so
    //   the base color of an emitting material is dropped.
    // - A material with an emissive texture does not emit. Textures are not
    //   read yet, and the factor alone would light up the whole surface
    //   (DamagedHelmet: factor 1, texture black except for a few lamps).
    int materialFor(int gltfMaterial)
    {
        if (materialOverride >= 0)
        {
            return materialOverride;
        }
        auto found = materialIds.find(gltfMaterial);
        if (found != materialIds.end())
        {
            return found->second;
        }
        Material m{};
        m.type = DIFFUSE;
        m.color = glm::vec3(1.0f);
        if (gltfMaterial >= 0 && gltfMaterial < (int)model.materials.size())
        {
            const tinygltf::Material& source = model.materials[gltfMaterial];
            const std::vector<double>& base = source.pbrMetallicRoughness.baseColorFactor;
            if (base.size() >= 3)
            {
                m.color = glm::vec3((float)base[0], (float)base[1], (float)base[2]);
            }
            const std::vector<double>& e = source.emissiveFactor;
            const glm::vec3 emissive = e.size() >= 3 ? glm::vec3((float)e[0], (float)e[1], (float)e[2]) : glm::vec3(0.0f);
            const float strength = extensionNumber(source, "KHR_materials_emissive_strength", "emissiveStrength", 1.0f);
            const bool textured = source.emissiveTexture.index >= 0;
            if (!textured && strength > 0.0f && glm::max(emissive.r, glm::max(emissive.g, emissive.b)) > 0.0f)
            {
                m.type = EMISSIVE;
                m.color = emissive;
                m.emittance = strength;
            }
        }
        const int id = (int)scene.materials.size();
        scene.materials.push_back(m);
        materialIds[gltfMaterial] = id;
        return id;
    }

    // Keeps the pose of the first perspective camera the walk reaches; a
    // file's other cameras are ignored.
    void cameraFor(int gltfCamera, const glm::mat4& world)
    {
        if (camera || gltfCamera < 0 || gltfCamera >= (int)model.cameras.size())
        {
            return;
        }
        const tinygltf::Camera& source = model.cameras[gltfCamera];
        // The upper 3x3 holds the camera's axes. Its determinant is negative
        // when the transform mirrors, and zero when a scale of 0 has
        // flattened the axes, which leaves no direction to look in.
        const float det = glm::determinant(glm::mat3(world));
        if (source.type != "perspective" || source.perspective.yfov <= 0.0 || det == 0.0f)
        {
            fprintf(stderr, "glTF %s: camera %d skipped: not a usable perspective camera\n", path.c_str(), gltfCamera);
            return;
        }
        GltfCamera c;
        c.position = glm::vec3(world[3]);
        c.view = -glm::normalize(glm::vec3(world[2]));
        c.up = glm::normalize(glm::vec3(world[1]));
        c.mirrored = det < 0.0f;
        c.yfov = (float)source.perspective.yfov;
        c.aspectRatio = (float)source.perspective.aspectRatio;
        camera = c;
    }

    // Appends the primitive's vertices and triangles to the scene's flat
    // arrays and returns its TriangleMesh index, or -1 if it cannot be used.
    int meshFor(int meshIndex, int primIndex)
    {
        const std::pair<int, int> key(meshIndex, primIndex);
        auto found = meshIds.find(key);
        if (found != meshIds.end())
        {
            return found->second;
        }
        const int id = loadPrimitive(meshIndex, primIndex);
        meshIds[key] = id;
        return id;
    }

    int loadPrimitive(int meshIndex, int primIndex)
    {
        const tinygltf::Primitive& prim = model.meshes[meshIndex].primitives[primIndex];
        if (prim.mode != TINYGLTF_MODE_TRIANGLES)
        {
            warn("not a triangle list (strips and fans are not supported)", meshIndex, primIndex);
            return -1;
        }
        std::string err;

        // Positions: required, float VEC3.
        auto posAttr = prim.attributes.find("POSITION");
        AccessorView pos;
        if (posAttr == prim.attributes.end() || !viewAccessor(model, posAttr->second, pos, err) || !isFloatVec3(pos))
        {
            warn(err.empty() ? "no float POSITION attribute" : err.c_str(), meshIndex, primIndex);
            return -1;
        }
        const int baseVertex = (int)scene.positions.size();
        for (size_t i = 0; i < pos.count; ++i)
        {
            scene.positions.push_back(readVec3(pos, i));
        }

        // Indices: from the accessor, or 0, 1, 2, ... when the primitive is
        // not indexed. Stored offset by baseVertex, so they address the
        // scene's flat arrays directly.
        std::vector<unsigned int> local;
        if (prim.indices >= 0)
        {
            AccessorView idx;
            if (!viewAccessor(model, prim.indices, idx, err) || idx.numComponents != 1 || !isIndexType(idx.componentType))
            {
                warn(err.empty() ? "unsupported index accessor" : err.c_str(), meshIndex, primIndex);
                scene.positions.resize(baseVertex);
                return -1;
            }
            local.resize(idx.count);
            for (size_t i = 0; i < idx.count; ++i)
            {
                local[i] = readIndex(idx, i);
            }
        }
        else
        {
            local.resize(pos.count);
            for (size_t i = 0; i < pos.count; ++i)
            {
                local[i] = (unsigned int)i;
            }
        }
        if (local.size() < 3 || local.size() % 3 != 0)
        {
            warn("index count is not a positive multiple of three", meshIndex, primIndex);
            scene.positions.resize(baseVertex);
            return -1;
        }
        const int triCount = (int)(local.size() / 3);
        const int indexOffset = (int)scene.indices.size();
        for (int t = 0; t < triCount; ++t)
        {
            const unsigned int a = local[3 * t];
            const unsigned int b = local[3 * t + 1];
            const unsigned int c = local[3 * t + 2];
            if (a >= pos.count || b >= pos.count || c >= pos.count)
            {
                warn("index out of range", meshIndex, primIndex);
                scene.positions.resize(baseVertex);
                scene.indices.resize(indexOffset);
                return -1;
            }
            scene.indices.push_back(glm::ivec3(baseVertex + a, baseVertex + b, baseVertex + c));
        }

        // Normals: from the file when present and well formed, else
        // area-weighted face normals accumulated per vertex (the cross
        // product's length is twice the triangle's area, so summing the raw
        // cross products weights big triangles more).
        auto nrmAttr = prim.attributes.find("NORMAL");
        AccessorView nrm;
        if (nrmAttr != prim.attributes.end() && viewAccessor(model, nrmAttr->second, nrm, err)
            && isFloatVec3(nrm) && nrm.count == pos.count)
        {
            for (size_t i = 0; i < nrm.count; ++i)
            {
                scene.normals.push_back(readVec3(nrm, i));
            }
        }
        else
        {
            std::vector<glm::vec3> sum(pos.count, glm::vec3(0.0f));
            for (int t = 0; t < triCount; ++t)
            {
                const glm::ivec3 tri = scene.indices[indexOffset + t] - glm::ivec3(baseVertex);
                const glm::vec3 p0 = scene.positions[baseVertex + tri.x];
                const glm::vec3 p1 = scene.positions[baseVertex + tri.y];
                const glm::vec3 p2 = scene.positions[baseVertex + tri.z];
                const glm::vec3 n = glm::cross(p1 - p0, p2 - p0);
                sum[tri.x] += n;
                sum[tri.y] += n;
                sum[tri.z] += n;
            }
            for (size_t i = 0; i < pos.count; ++i)
            {
                // A vertex no triangle uses keeps a placeholder; nothing reads it.
                scene.normals.push_back(glm::dot(sum[i], sum[i]) > 0.0f ? glm::normalize(sum[i]) : glm::vec3(0.0f, 0.0f, 1.0f));
            }
        }

        TriangleMesh mesh;
        mesh.indexOffset = indexOffset;
        mesh.triCount = triCount;
        scene.meshes.push_back(mesh);
        return (int)scene.meshes.size() - 1;
    }

    // Walks the node tree, accumulating transforms, and makes one Geom per
    // primitive of every node that has a mesh.
    void visit(int nodeIndex, const glm::mat4& parent, int depth)
    {
        if (nodeIndex < 0 || nodeIndex >= (int)model.nodes.size() || depth > 64)
        {
            return;  // a malformed file: bad reference or a cycle
        }
        const tinygltf::Node& node = model.nodes[nodeIndex];
        const glm::mat4 world = parent * nodeLocalMatrix(node);
        cameraFor(node.camera, world);
        if (node.mesh >= 0 && node.mesh < (int)model.meshes.size())
        {
            const tinygltf::Mesh& mesh = model.meshes[node.mesh];
            for (size_t p = 0; p < mesh.primitives.size(); ++p)
            {
                const int meshId = meshFor(node.mesh, (int)p);
                if (meshId < 0)
                {
                    continue;
                }
                // translation / rotation / scale stay zero: the matrix holds
                // the whole transform and nothing reads the parts.
                Geom g{};
                g.type = MESH;
                g.materialid = materialFor(mesh.primitives[p].material);
                g.meshId = meshId;
                g.transform = world;
                g.inverseTransform = glm::inverse(world);
                g.invTranspose = glm::inverseTranspose(world);
                scene.geoms.push_back(g);
            }
        }
        for (int child : node.children)
        {
            visit(child, world, depth + 1);
        }
    }
};
}  // namespace

bool loadGltf(const std::string& path, const glm::mat4& sceneTransform, int materialOverride, Scene& scene,
    GltfInfo* info)
{
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    loader.SetImageLoader(skipImage, nullptr);
    std::string err;
    std::string warn;
    std::string ext = path.size() >= 4 ? path.substr(path.size() - 4) : "";
    for (char& c : ext)
    {
        c = (char)tolower((unsigned char)c);
    }
    const bool binary = ext == ".glb";
    const bool loaded = binary ? loader.LoadBinaryFromFile(&model, &err, &warn, path)
                               : loader.LoadASCIIFromFile(&model, &err, &warn, path);
    if (!warn.empty())
    {
        fprintf(stderr, "glTF %s: %s", path.c_str(), warn.c_str());
    }
    if (!loaded)
    {
        fprintf(stderr, "glTF %s: %s\n", path.c_str(), err.empty() ? "failed to load" : err.c_str());
        return false;
    }

    Loader l{ model, path, scene, materialOverride };
    const size_t firstGeom = scene.geoms.size();
    const size_t firstMesh = scene.meshes.size();
    const size_t firstTriangle = scene.indices.size();

    // Start from the default scene's root nodes. A file with no scenes lists
    // nodes only, so every node nobody names as a child is a root.
    const int sceneIndex = model.defaultScene >= 0 ? model.defaultScene : 0;
    if (sceneIndex < (int)model.scenes.size())
    {
        for (int root : model.scenes[sceneIndex].nodes)
        {
            l.visit(root, sceneTransform, 0);
        }
    }
    else
    {
        std::vector<bool> isChild(model.nodes.size(), false);
        for (const tinygltf::Node& node : model.nodes)
        {
            for (int child : node.children)
            {
                if (child >= 0 && child < (int)isChild.size())
                {
                    isChild[child] = true;
                }
            }
        }
        for (size_t i = 0; i < model.nodes.size(); ++i)
        {
            if (!isChild[i])
            {
                l.visit((int)i, sceneTransform, 0);
            }
        }
    }

    // World-space bounds of what was added, for placing the object in a scene.
    glm::vec3 lo(FLT_MAX);
    glm::vec3 hi(-FLT_MAX);
    for (size_t g = firstGeom; g < scene.geoms.size(); ++g)
    {
        const Geom& geom = scene.geoms[g];
        const TriangleMesh& mesh = scene.meshes[geom.meshId];
        for (int t = 0; t < mesh.triCount; ++t)
        {
            const glm::ivec3 tri = scene.indices[mesh.indexOffset + t];
            for (int k = 0; k < 3; ++k)
            {
                const glm::vec3 p = glm::vec3(geom.transform * glm::vec4(scene.positions[tri[k]], 1.0f));
                lo = glm::min(lo, p);
                hi = glm::max(hi, p);
            }
        }
    }
    printf("glTF %s: %zu triangles in %zu primitives, %zu instances, bounds (%.3f, %.3f, %.3f) to (%.3f, %.3f, %.3f)\n",
        path.c_str(), scene.indices.size() - firstTriangle, scene.meshes.size() - firstMesh,
        scene.geoms.size() - firstGeom, lo.x, lo.y, lo.z, hi.x, hi.y, hi.z);
    reportUnusedLights(model, path);

    if (info != nullptr)
    {
        info->boundsMin = lo;
        info->boundsMax = hi;
        info->camera = l.camera;
        info->render = renderHints(model);
    }
    return true;
}
