#pragma once

#include <cuda_runtime.h>

#include "glm/glm.hpp"

#include <string>
#include <vector>

enum class GeomType
{
    SPHERE,
    CUBE,
    MESH  // one glTF primitive's triangles, see TriangleMesh
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

// One glTF primitive: triCount consecutive triangles of the scene's flat
// index array. Each index points into the flat vertex arrays (positions,
// normals, uvs and tangents share it), already offset, so a triangle needs no
// base vertex. A Geom of type MESH instances one of these under its
// transform; several Geoms may share one (a mesh used by more than one glTF
// node).
struct TriangleMesh
{
    int indexOffset;  // first triangle in the index array
    int triCount;
};

// Device pointers to the scene's flat mesh arrays, uploaded once by
// pathtraceInit and read by both intersection paths (the naive kernel loops
// over them; OptiX builds its acceleration structures from them and reads the
// vertex attributes in its hit programs). Read-only after the upload.
struct MeshBuffers
{
    glm::vec3* positions;
    glm::vec3* normals;     // per vertex, unit length, object space
    glm::vec2* uvs;         // per vertex, TEXCOORD_0; (0, 0) for a primitive without one
    // per vertex, object space: xyz along increasing u, w = +-1 the bitangent
    // sign (bitangent = w * cross(normal, tangent), as in glTF); all zero for
    // a primitive whose material has no normal map
    glm::vec4* tangents;
    glm::ivec3* indices;    // per triangle, into the vertex arrays
    TriangleMesh* meshes;   // indexed by Geom::meshId
};

// A decoded glTF image: 8-bit RGBA, rows top to bottom. glTF puts uv (0, 0)
// at the top left of the image, so a texture fetch at uv reads it as stored.
struct TextureImage
{
    int width;
    int height;
    std::vector<unsigned char> rgba;  // width * height * 4
};

// One image as a material slot reads it: the glTF sampler's wrap and filter
// modes, and whether the texels are sRGB-encoded color, which the texture
// unit decodes to linear, or linear data. Becomes one CUDA texture object
// (textures.cpp); several may share an image.
struct Texture
{
    int image;  // into Scene::textureImages
    cudaTextureAddressMode wrapU;
    cudaTextureAddressMode wrapV;
    cudaTextureFilterMode filter;
    bool srgb;
};

struct Geom
{
    GeomType type;
    int materialId;
    int meshId;  // MESH only: which TriangleMesh; -1 for the primitives
    // MESH only: -1 when transform mirrors (negative determinant), which
    // flips the bitangent sign of the mesh's tangents; 1 otherwise
    float tangentSign;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
};

// glTF's alpha modes. BLEND is loaded as OPAQUE.
enum AlphaMode
{
    ALPHA_OPAQUE,
    ALPHA_MASK  // a hit whose alpha is below alphaCutoff is no hit
};

// Every material is glTF's metallic-roughness model with the extensions
// bsdf.h lists, shaded by scatterPbr. A glTF material fills it from the file
// (gltf_loader.cpp); a scene JSON's material types are translated into it
// (scene.cpp). The defaults are glTF's default material: white, fully
// metallic, fully rough. Every texture slot indexes Scene::textures (-1 for
// none), and the texture value multiplies the factor it belongs to.
struct Material
{
    glm::vec3 baseColor = glm::vec3(1.0f);        // base color factor
    float ior = 1.5f;                             // KHR_materials_ior; 0 means a Fresnel term of 1
    float alpha = 1.0f;                           // base color factor alpha
    AlphaMode alphaMode = ALPHA_OPAQUE;
    float alphaCutoff = 0.5f;
    float metallic = 1.0f;
    float roughness = 1.0f;
    float normalScale = 1.0f;                     // scales the normal map's x and y
    glm::vec3 emission = glm::vec3(0.0f);         // emissive factor * KHR_materials_emissive_strength
    float transmission = 0.0f;                    // KHR_materials_transmission
    float specularFactor = 1.0f;                  // KHR_materials_specular
    glm::vec3 specularColorFactor = glm::vec3(1.0f);
    float clearcoat = 0.0f;                       // KHR_materials_clearcoat
    float clearcoatRoughness = 0.0f;
    // KHR_materials_volume: absorption per unit of world distance inside the
    // object, -ln(attenuationColor) / attenuationDistance; zero without it
    glm::vec3 absorption = glm::vec3(0.0f);
    int baseColorTexture = -1;                    // sRGB rgb, linear alpha
    int metallicRoughnessTexture = -1;            // G roughness, B metallic
    int normalTexture = -1;                       // tangent space, +Y up
    int emissiveTexture = -1;                     // sRGB
    int transmissionTexture = -1;                 // R
};

struct Camera
{
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 pixelLength;
    // right is -cross(view, up) instead of cross(view, up), so the image is
    // flipped left to right. Set by a glTF camera whose transform mirrors.
    bool mirrored;
};

struct RenderState
{
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment
{
    Ray ray;
    glm::vec3 color;
    int pixelIndex;
    int remainingBounces;
    // The material the path is inside, -1 in air: set when it refracts into
    // a PBR surface, cleared when it refracts out. Its KHR_materials_volume
    // absorption applies to every segment in between.
    int medium;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
  float t;
  glm::vec3 surfaceNormal;
  glm::vec2 uv;  // texture coordinates at the hit; (0, 0) on spheres and cubes
  // World-space tangent at the hit (xyz, not unit length) and bitangent sign
  // (w), see MeshBuffers::tangents. The sign already includes a mirroring
  // instance transform. All zero on spheres, cubes and meshes without one.
  glm::vec4 tangent;
  int materialId;
  bool outside;
};
