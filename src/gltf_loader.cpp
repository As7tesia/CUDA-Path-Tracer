// glTF loading through tinygltf. See gltf_loader.h.
//
// This is the one translation unit that compiles tinygltf's implementation.
// The repo already has nlohmann's json.hpp, so tinygltf uses that instead of
// its own copy. Images are decoded by tinygltf's built-in loader through
// stb_image (implementation compiled in stb.cpp), whether they are external
// files, data URIs or buffer views: every image comes out as RGBA, 8 or 16
// bits per channel. An image file that is missing is a warning and the image
// stays empty; one that exists and does not decode fails the whole load.
// The glTF writer is off.

#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_INCLUDE_JSON
#define TINYGLTF_NO_STB_IMAGE_WRITE
#include "json.hpp"
#include "tiny_gltf.h"

#include "gltf_loader.h"
#include "scene.h"
#include "utilities.h"

#include "mikktspace.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include <cctype>
#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace
{
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

// Texture coordinates come as floats, or as unsigned bytes or shorts that
// glTF requires to be normalized to [0, 1]; readUv assumes one of the three.
bool isUvType(int componentType)
{
    return componentType == TINYGLTF_COMPONENT_TYPE_FLOAT
        || componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE
        || componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT;
}

glm::vec2 readUv(const AccessorView& v, size_t i)
{
    const unsigned char* p = v.at(i);
    switch (v.componentType)
    {
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
        return glm::vec2(p[0], p[1]) / 255.0f;
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
    {
        const uint16_t* s = reinterpret_cast<const uint16_t*>(p);
        return glm::vec2(s[0], s[1]) / 65535.0f;
    }
    default:
    {
        const float* f = reinterpret_cast<const float*>(p);
        return glm::vec2(f[0], f[1]);
    }
    }
}

// glTF sampler wrap mode to CUDA's. REPEAT is glTF's default.
cudaTextureAddressMode addressMode(int gltfWrap)
{
    switch (gltfWrap)
    {
    case TINYGLTF_TEXTURE_WRAP_CLAMP_TO_EDGE:
        return cudaAddressModeClamp;
    case TINYGLTF_TEXTURE_WRAP_MIRRORED_REPEAT:
        return cudaAddressModeMirror;
    default:
        return cudaAddressModeWrap;
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

// One of a material's extensions as a JSON object, or a null value when the
// material does not have it.
const tinygltf::Value& extension(const tinygltf::Material& material, const char* name)
{
    static const tinygltf::Value null;
    auto found = material.extensions.find(name);
    return found == material.extensions.end() ? null : found->second;
}

// A number from one of a material's extensions, or fallback when the
// material does not have the extension or the extension does not say.
float extensionNumber(const tinygltf::Material& material, const char* name, const char* key, float fallback)
{
    return numberMember(extension(material, name), key, fallback);
}

// A color (three numbers) from one of a material's extensions, or fallback.
glm::vec3 extensionColor(const tinygltf::Material& material, const char* name, const char* key, glm::vec3 fallback)
{
    const tinygltf::Value& value = member(extension(material, name), key);
    if (value.ArrayLen() != 3)
    {
        return fallback;
    }
    glm::vec3 color;
    for (int i = 0; i < 3; ++i)
    {
        const tinygltf::Value& c = value.Get(i);
        if (!c.IsNumber())
        {
            return fallback;
        }
        color[i] = (float)c.GetNumberAsDouble();
    }
    return color;
}

// A texture slot inside one of a material's extensions ({"index", "texCoord"}),
// in the form tinygltf gives the core slots; index -1 when there is none.
tinygltf::TextureInfo extensionTexture(const tinygltf::Material& material, const char* name, const char* key)
{
    const tinygltf::Value& slot = member(extension(material, name), key);
    tinygltf::TextureInfo info;
    info.index = (int)numberMember(slot, "index", -1.0f);
    info.texCoord = (int)numberMember(slot, "texCoord", 0.0f);
    return info;
}

// The material extensions this loader reads. Others are named once per file.
const char* const kReadExtensions[] = {
    "KHR_materials_emissive_strength",
    "KHR_materials_ior",
    "KHR_materials_transmission",
    "KHR_materials_volume",
    "KHR_materials_specular",
    "KHR_materials_clearcoat",
};

bool isReadExtension(const std::string& name)
{
    for (const char* read : kReadExtensions)
    {
        if (name == read)
        {
            return true;
        }
    }
    return false;
}

// Names, once per file, what the materials ask for that the loader does not
// do, so a render that differs from the asset's look explains itself. The
// occlusion texture is left out on purpose and not named: it stands in for
// the shadowing a path tracer computes anyway.
void reportUnsupportedMaterialFeatures(const tinygltf::Model& model, const std::string& path)
{
    std::map<std::string, int> extensions;
    int blend = 0;
    for (const tinygltf::Material& material : model.materials)
    {
        // Counted once per material, however many of its slots carry them.
        std::set<std::string> unread;
        for (const auto& e : material.extensions)
        {
            if (!isReadExtension(e.first))
            {
                unread.insert(e.first);
            }
        }
        // Texture-level extensions (KHR_texture_transform) sit on the slots
        // the loader reads, not on the material; none of them is applied.
        const tinygltf::ExtensionMap* slots[] = {
            &material.pbrMetallicRoughness.baseColorTexture.extensions,
            &material.pbrMetallicRoughness.metallicRoughnessTexture.extensions,
            &material.normalTexture.extensions,
            &material.emissiveTexture.extensions,
        };
        for (const tinygltf::ExtensionMap* slot : slots)
        {
            for (const auto& e : *slot)
            {
                unread.insert(e.first);
            }
        }
        const tinygltf::Value& transmissionSlot =
            member(member(extension(material, "KHR_materials_transmission"), "transmissionTexture"), "extensions");
        if (transmissionSlot.IsObject())
        {
            for (const std::string& key : transmissionSlot.Keys())
            {
                unread.insert(key);
            }
        }
        for (const std::string& name : unread)
        {
            ++extensions[name];
        }
        blend += material.alphaMode == "BLEND";
    }
    for (const auto& e : extensions)
    {
        fprintf(stderr, "glTF %s: %s ignored (%d materials)\n", path.c_str(), e.first.c_str(), e.second);
    }
    if (blend > 0)
    {
        fprintf(stderr, "glTF %s: alphaMode BLEND rendered opaque (%d materials)\n", path.c_str(), blend);
    }
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

// One primitive as MikkTSpace sees it: triangles over the primitive's own
// vertices, and a tangent out for every triangle corner.
struct MikkPrimitive
{
    const glm::vec3* positions;
    const glm::vec3* normals;
    const glm::vec2* uvs;
    const std::vector<unsigned int>& indices;  // three per triangle
    std::vector<glm::vec4> corners;            // tangent and sign per corner, filled by mikkSetTangent

    unsigned int vertex(int face, int vert) const { return indices[face * 3 + vert]; }
};

const MikkPrimitive& mikkData(const SMikkTSpaceContext* context)
{
    return *static_cast<const MikkPrimitive*>(context->m_pUserData);
}

int mikkNumFaces(const SMikkTSpaceContext* context)
{
    return (int)(mikkData(context).indices.size() / 3);
}

int mikkNumVerticesOfFace(const SMikkTSpaceContext*, const int)
{
    return 3;
}

void mikkPosition(const SMikkTSpaceContext* context, float out[], const int face, const int vert)
{
    const MikkPrimitive& p = mikkData(context);
    const glm::vec3 v = p.positions[p.vertex(face, vert)];
    out[0] = v.x;
    out[1] = v.y;
    out[2] = v.z;
}

void mikkNormal(const SMikkTSpaceContext* context, float out[], const int face, const int vert)
{
    const MikkPrimitive& p = mikkData(context);
    const glm::vec3 n = p.normals[p.vertex(face, vert)];
    out[0] = n.x;
    out[1] = n.y;
    out[2] = n.z;
}

// glTF's v runs down the image, while MikkTSpace (and Blender, where most
// normal maps are baked against it) take v up. MikkTSpace's bitangent follows
// increasing v, and glTF wants it toward the top of the image (the normal
// map's +Y), so it gets 1 - v. The tangent follows u and is the same either
// way; only the sign changes.
void mikkTexCoord(const SMikkTSpaceContext* context, float out[], const int face, const int vert)
{
    const MikkPrimitive& p = mikkData(context);
    const glm::vec2 uv = p.uvs[p.vertex(face, vert)];
    out[0] = uv.x;
    out[1] = 1.0f - uv.y;
}

void mikkSetTangent(const SMikkTSpaceContext* context, const float tangent[], const float sign, const int face,
    const int vert)
{
    MikkPrimitive& p = *static_cast<MikkPrimitive*>(context->m_pUserData);
    p.corners[face * 3 + vert] = glm::vec4(tangent[0], tangent[1], tangent[2], sign < 0.0f ? -1.0f : 1.0f);
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
    std::map<std::pair<int, bool>, int> textureIds;    // (glTF texture, sRGB) -> Scene texture, -1 if unusable
    std::map<int, int> imageIds;                       // glTF image -> Scene texture image, -1 if unusable
    std::optional<GltfCamera> camera;                  // the first usable camera the walk reached

    // A primitive this loader cannot use is skipped, not fatal: the rest of
    // the file still loads.
    void warn(const char* what, int meshIndex, int primIndex)
    {
        fprintf(stderr, "glTF %s: mesh %d primitive %d skipped: %s\n", path.c_str(), meshIndex, primIndex, what);
    }

    // Every glTF material becomes a PBR material (bsdf.h): the core
    // metallic-roughness factors and texture slots (base color,
    // metallic-roughness, normal, emissive), the alpha mode, and the factors
    // of the extensions in kReadExtensions. Of the extensions' textures only
    // transmissionTexture is read. A primitive without a material gets glTF's
    // default material: white, fully metallic, fully rough.
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
        m.type = PBR;
        m.color = glm::vec3(1.0f);
        m.ior = 1.5f;
        if (gltfMaterial >= 0 && gltfMaterial < (int)model.materials.size())
        {
            const tinygltf::Material& source = model.materials[gltfMaterial];
            const tinygltf::PbrMetallicRoughness& pbr = source.pbrMetallicRoughness;
            if (pbr.baseColorFactor.size() == 4)
            {
                m.color = glm::vec3((float)pbr.baseColorFactor[0], (float)pbr.baseColorFactor[1], (float)pbr.baseColorFactor[2]);
                m.alpha = (float)pbr.baseColorFactor[3];
            }
            m.metallic = (float)pbr.metallicFactor;
            m.roughness = (float)pbr.roughnessFactor;
            m.baseColorTexture = textureFor(pbr.baseColorTexture.index, pbr.baseColorTexture.texCoord, true);
            m.metallicRoughnessTexture = textureFor(pbr.metallicRoughnessTexture.index, pbr.metallicRoughnessTexture.texCoord, false);
            m.normalTexture = textureFor(source.normalTexture.index, source.normalTexture.texCoord, false);
            m.normalScale = (float)source.normalTexture.scale;

            // Emission is the emissive factor times the emissive texture
            // times KHR_materials_emissive_strength (1 without it). A black
            // factor leaves the texture nothing to scale.
            const std::vector<double>& e = source.emissiveFactor;
            const glm::vec3 emissive = e.size() == 3 ? glm::vec3((float)e[0], (float)e[1], (float)e[2]) : glm::vec3(0.0f);
            m.emission = emissive * extensionNumber(source, "KHR_materials_emissive_strength", "emissiveStrength", 1.0f);
            if (maxComponent(m.emission) > 0.0f)
            {
                m.emissiveTexture = textureFor(source.emissiveTexture.index, source.emissiveTexture.texCoord, true);
            }

            if (source.alphaMode == "MASK")
            {
                m.alphaMode = ALPHA_MASK;
                m.alphaCutoff = (float)source.alphaCutoff;
            }

            // ior 0 is glTF's code for a Fresnel term of 1; any other value
            // below 1 is invalid and read as 1.
            m.ior = extensionNumber(source, "KHR_materials_ior", "ior", 1.5f);
            if (m.ior != 0.0f && m.ior < 1.0f)
            {
                m.ior = 1.0f;
            }

            m.transmission = glm::clamp(extensionNumber(source, "KHR_materials_transmission", "transmissionFactor", 0.0f), 0.0f, 1.0f);
            if (m.transmission > 0.0f)
            {
                const tinygltf::TextureInfo slot = extensionTexture(source, "KHR_materials_transmission", "transmissionTexture");
                m.transmissionTexture = textureFor(slot.index, slot.texCoord, false);
            }

            // KHR_materials_volume contributes only its absorption: every
            // transmissive surface is a solid boundary here, so the
            // thickness that tells thin-walled from solid is not read. An
            // absent attenuation distance is infinite, no absorption.
            const float distance = extensionNumber(source, "KHR_materials_volume", "attenuationDistance", 0.0f);
            if (distance > 0.0f)
            {
                const glm::vec3 color = extensionColor(source, "KHR_materials_volume", "attenuationColor", glm::vec3(1.0f));
                m.absorption = -glm::log(glm::clamp(color, glm::vec3(1e-6f), glm::vec3(1.0f))) / distance;
            }

            m.specularFactor = glm::clamp(extensionNumber(source, "KHR_materials_specular", "specularFactor", 1.0f), 0.0f, 1.0f);
            m.specularColorFactor = glm::max(extensionColor(source, "KHR_materials_specular", "specularColorFactor", glm::vec3(1.0f)),
                glm::vec3(0.0f));
            m.clearcoat = glm::clamp(extensionNumber(source, "KHR_materials_clearcoat", "clearcoatFactor", 0.0f), 0.0f, 1.0f);
            m.clearcoatRoughness = glm::clamp(
                extensionNumber(source, "KHR_materials_clearcoat", "clearcoatRoughnessFactor", 0.0f), 0.0f, 1.0f);
        }
        const int id = (int)scene.materials.size();
        scene.materials.push_back(m);
        materialIds[gltfMaterial] = id;
        return id;
    }

    // The Scene texture for one of a material's texture slots (glTF texture
    // index and texture coordinate set), or -1 when the slot is empty or its
    // image is unusable; the material then uses its factor alone. srgb says
    // the slot holds color (base color, emissive) rather than data. Only
    // TEXCOORD_0 is loaded, so a slot that names another set is read with
    // TEXCOORD_0.
    int textureFor(int index, int texCoord, bool srgb)
    {
        if (index < 0)
        {
            return -1;
        }
        const std::pair<int, bool> key(index, srgb);
        auto found = textureIds.find(key);
        if (found != textureIds.end())
        {
            return found->second;
        }
        if (texCoord != 0)
        {
            fprintf(stderr, "glTF %s: texture %d reads TEXCOORD_%d, only TEXCOORD_0 is loaded\n",
                path.c_str(), index, texCoord);
        }
        int id = -1;
        const tinygltf::Texture* source = index < (int)model.textures.size() ? &model.textures[index] : nullptr;
        const int image = imageFor(source != nullptr ? source->source : -1);
        if (image >= 0)
        {
            Texture t;
            t.image = image;
            t.wrapU = cudaAddressModeWrap;
            t.wrapV = cudaAddressModeWrap;
            t.filter = cudaFilterModeLinear;
            t.srgb = srgb;
            if (source->sampler >= 0 && source->sampler < (int)model.samplers.size())
            {
                const tinygltf::Sampler& sampler = model.samplers[source->sampler];
                t.wrapU = addressMode(sampler.wrapS);
                t.wrapV = addressMode(sampler.wrapT);
                // There are no mipmaps, so only the magnification filter
                // applies. Minification needs none: the camera jitters its
                // rays across the pixel, so samples average the texels a
                // pixel covers.
                t.filter = sampler.magFilter == TINYGLTF_TEXTURE_FILTER_NEAREST ? cudaFilterModePoint : cudaFilterModeLinear;
            }
            id = (int)scene.textures.size();
            scene.textures.push_back(t);
        }
        textureIds[key] = id;
        return id;
    }

    // The Scene image for a glTF image, or -1 when tinygltf has no pixels for
    // it: the file is missing, or there is no image index (-1), which is what
    // a texture whose image only an extension names (KTX2, WebP) has, as does
    // a bad texture index. tinygltf decodes to RGBA; 16-bit images are
    // rounded to 8 bits, (v + 128) / 257 mapping 0..65535 onto 0..255.
    int imageFor(int gltfImage)
    {
        auto found = imageIds.find(gltfImage);
        if (found != imageIds.end())
        {
            return found->second;
        }
        int id = -1;
        const tinygltf::Image* source =
            gltfImage >= 0 && gltfImage < (int)model.images.size() ? &model.images[gltfImage] : nullptr;
        const bool usable = source != nullptr && source->width > 0 && source->height > 0 && source->component == 4
            && (source->bits == 8 || source->bits == 16)
            && source->image.size() == (size_t)source->width * source->height * 4 * (source->bits / 8);
        if (!usable)
        {
            fprintf(stderr, "glTF %s: image %d has no pixels, textures using it are skipped\n", path.c_str(), gltfImage);
        }
        else
        {
            TextureImage image;
            image.width = source->width;
            image.height = source->height;
            if (source->bits == 8)
            {
                image.rgba = source->image;
            }
            else
            {
                image.rgba.resize((size_t)image.width * image.height * 4);
                const uint16_t* wide = reinterpret_cast<const uint16_t*>(source->image.data());
                for (size_t i = 0; i < image.rgba.size(); ++i)
                {
                    image.rgba[i] = (unsigned char)((wide[i] + 128) / 257);
                }
            }
            id = (int)scene.textureImages.size();
            scene.textureImages.push_back(std::move(image));
        }
        imageIds[gltfImage] = id;
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

        // Texture coordinates: TEXCOORD_0 when present and well formed, else
        // (0, 0) at every vertex, which only matters to a textured material.
        auto uvAttr = prim.attributes.find("TEXCOORD_0");
        AccessorView uv;
        std::string uvErr;
        bool hasUv = false;
        if (uvAttr != prim.attributes.end() && viewAccessor(model, uvAttr->second, uv, uvErr)
            && uv.numComponents == 2 && isUvType(uv.componentType) && uv.count == pos.count)
        {
            hasUv = true;
            for (size_t i = 0; i < uv.count; ++i)
            {
                scene.uvs.push_back(readUv(uv, i));
            }
        }
        else
        {
            if (uvAttr != prim.attributes.end())
            {
                fprintf(stderr, "glTF %s: mesh %d primitive %d: TEXCOORD_0 unusable (%s), uvs set to 0\n", path.c_str(),
                    meshIndex, primIndex, uvErr.empty() ? "unsupported layout" : uvErr.c_str());
            }
            scene.uvs.resize(scene.uvs.size() + pos.count, glm::vec2(0.0f));
        }

        // Tangents, which only a normal map uses: TANGENT when present and
        // well formed, else generated when the primitive's material has a
        // normal map and the primitive has texture coordinates to generate
        // them from, else zero (the shade kernel skips the map then).
        auto tanAttr = prim.attributes.find("TANGENT");
        AccessorView tan;
        std::string tanErr;
        if (tanAttr != prim.attributes.end() && viewAccessor(model, tanAttr->second, tan, tanErr)
            && tan.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT && tan.numComponents == 4 && tan.count == pos.count)
        {
            for (size_t i = 0; i < tan.count; ++i)
            {
                const float* f = reinterpret_cast<const float*>(tan.at(i));
                scene.tangents.push_back(glm::vec4(f[0], f[1], f[2], f[3] < 0.0f ? -1.0f : 1.0f));
            }
        }
        else
        {
            if (tanAttr != prim.attributes.end())
            {
                fprintf(stderr, "glTF %s: mesh %d primitive %d: TANGENT unusable (%s), generating it\n", path.c_str(),
                    meshIndex, primIndex, tanErr.empty() ? "unsupported layout" : tanErr.c_str());
            }
            const bool normalMapped = hasUv && materialOverride < 0 && prim.material >= 0 && prim.material < (int)model.materials.size()
                && model.materials[prim.material].normalTexture.index >= 0;
            if (normalMapped)
            {
                generateTangents(baseVertex, pos.count, indexOffset, triCount);
            }
            else
            {
                scene.tangents.resize(scene.tangents.size() + pos.count, glm::vec4(0.0f));
            }
        }

        TriangleMesh mesh;
        mesh.indexOffset = indexOffset;
        mesh.triCount = triCount;
        scene.meshes.push_back(mesh);
        return (int)scene.meshes.size() - 1;
    }

    // Tangents for a primitive whose file has none, the way glTF asks for
    // them: MikkTSpace over the positions, normals and TEXCOORD_0, which is
    // what normal maps are baked against. The primitive's vertices start at
    // baseVertex and its triangles at indexOffset; positions, normals and
    // uvs are already in the scene, tangents are appended here.
    //
    // MikkTSpace gives every triangle corner its own tangent. Corners that
    // share a vertex nearly always agree; where they do not (a mirrored uv
    // seam, a hard break in the tangent frame), the vertex is copied so each
    // copy carries one tangent, and the corner is pointed at its copy.
    void generateTangents(int baseVertex, size_t vertexCount, int indexOffset, int triCount)
    {
        std::vector<unsigned int> local((size_t)triCount * 3);
        for (int t = 0; t < triCount; ++t)
        {
            for (int k = 0; k < 3; ++k)
            {
                local[3 * t + k] = (unsigned int)(scene.indices[indexOffset + t][k] - baseVertex);
            }
        }
        MikkPrimitive data{ scene.positions.data() + baseVertex, scene.normals.data() + baseVertex,
            scene.uvs.data() + baseVertex, local, std::vector<glm::vec4>(local.size(), glm::vec4(0.0f)) };
        SMikkTSpaceInterface callbacks = {};
        callbacks.m_getNumFaces = mikkNumFaces;
        callbacks.m_getNumVerticesOfFace = mikkNumVerticesOfFace;
        callbacks.m_getPosition = mikkPosition;
        callbacks.m_getNormal = mikkNormal;
        callbacks.m_getTexCoord = mikkTexCoord;
        callbacks.m_setTSpaceBasic = mikkSetTangent;
        SMikkTSpaceContext context = {};
        context.m_pInterface = &callbacks;
        context.m_pUserData = &data;
        if (!genTangSpaceDefault(&context))
        {
            fprintf(stderr, "glTF %s: MikkTSpace failed, normal map skipped for a primitive\n", path.c_str());
            scene.tangents.resize(scene.tangents.size() + vertexCount, glm::vec4(0.0f));
            return;
        }

        // Each vertex keeps a list of the tangents its corners have asked for
        // so far and which vertex (itself or a copy) carries each one.
        struct Variant
        {
            glm::vec4 tangent;
            unsigned int vertex;
            int next;  // the vertex's next variant, -1 at the end
        };
        std::vector<int> firstVariant(vertexCount, -1);
        std::vector<Variant> variants;
        std::vector<glm::vec4> tangents(vertexCount, glm::vec4(0.0f));
        for (size_t c = 0; c < local.size(); ++c)
        {
            const unsigned int v = local[c];
            const glm::vec4 tangent = data.corners[c];
            int k = firstVariant[v];
            while (k >= 0 && variants[k].tangent != tangent)
            {
                k = variants[k].next;
            }
            unsigned int carrier;
            if (k >= 0)
            {
                carrier = variants[k].vertex;
            }
            else
            {
                carrier = v;
                if (firstVariant[v] >= 0)
                {
                    // Copied through locals: push_back may reallocate.
                    const glm::vec3 position = scene.positions[baseVertex + v];
                    const glm::vec3 normal = scene.normals[baseVertex + v];
                    const glm::vec2 uv = scene.uvs[baseVertex + v];
                    scene.positions.push_back(position);
                    scene.normals.push_back(normal);
                    scene.uvs.push_back(uv);
                    carrier = (unsigned int)tangents.size();
                    tangents.push_back(glm::vec4(0.0f));
                }
                tangents[carrier] = tangent;
                variants.push_back({ tangent, carrier, firstVariant[v] });
                firstVariant[v] = (int)variants.size() - 1;
            }
            scene.indices[indexOffset + c / 3][(int)(c % 3)] = baseVertex + (int)carrier;
        }
        scene.tangents.insert(scene.tangents.end(), tangents.begin(), tangents.end());
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
                g.tangentSign = glm::determinant(glm::mat3(world)) < 0.0f ? -1.0f : 1.0f;
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
    glm::vec3 lo, hi;
    scene.bounds(firstGeom, scene.geoms.size(), lo, hi);
    printf("glTF %s: %zu triangles in %zu primitives, %zu instances, bounds (%.3f, %.3f, %.3f) to (%.3f, %.3f, %.3f)\n",
        path.c_str(), scene.indices.size() - firstTriangle, scene.meshes.size() - firstMesh,
        scene.geoms.size() - firstGeom, lo.x, lo.y, lo.z, hi.x, hi.y, hi.z);
    reportUnusedLights(model, path);
    if (materialOverride < 0)
    {
        reportUnsupportedMaterialFeatures(model, path);
    }

    if (info != nullptr)
    {
        info->boundsMin = lo;
        info->boundsMax = hi;
        info->camera = l.camera;
        info->render = renderHints(model);
    }
    return true;
}
