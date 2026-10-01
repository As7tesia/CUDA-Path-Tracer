// OptiX device programs for the intersection stage. Compiled by nvcc to
// OptiX-IR (see CMakeLists.txt), not linked like the other .cu files: the
// driver compiles them at run time when optixModuleCreate runs.
//
// The programs produce exactly what computeIntersections in pathtrace.cu
// produces, so shadeMaterial and the material sort cannot tell the two apart:
// t along the normalized ray, the surface normal with the same orientation
// convention as the naive tests, the uv, the material id, the outside flag,
// and the sort key. There is no payload: every program knows its launch
// index and writes straight into the intersection buffer.

#include <optix.h>

#include "optix_params.h"

extern "C"
{
    __constant__ OptixIntersectParams params;
}

static __forceinline__ __device__ float3 toFloat3(glm::vec3 v)
{
    return make_float3(v.x, v.y, v.z);
}

static __forceinline__ __device__ glm::vec3 toVec3(float3 v)
{
    return glm::vec3(v.x, v.y, v.z);
}

// One thread per path: read its ray and trace it. The direction is normalized
// so optixGetRayTmax() in the hit programs is a world-space distance, which is
// what shadeMaterial expects when it rebuilds the hit point as origin + t * dir.
extern "C" __global__ void __raygen__paths()
{
    const unsigned int i = optixGetLaunchIndex().x;
    const Ray ray = params.paths[i].ray;
    optixTrace(params.handle,
        toFloat3(ray.origin),
        toFloat3(glm::normalize(ray.direction)),
        0.0f,                  // tmin: the naive tests accept any t > 0 too
        1e16f,                 // tmax
        0.0f,                  // ray time, no motion blur
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        0,                     // SBT offset, added to the instance's sbtOffset
        1,                     // SBT stride: one ray type
        0);                    // miss program index
}

// A miss is t = -1 and the one-past-the-last material id as the sort key,
// the same as the naive kernel.
extern "C" __global__ void __miss__paths()
{
    const unsigned int i = optixGetLaunchIndex().x;
    params.intersections[i].t = -1.0f;
    params.materialIds[i] = params.numMaterials;
}

// The hit instance's record: material, and mesh for MESH instances.
static __forceinline__ __device__ InstanceRecord instance()
{
    return params.instances[optixGetInstanceId()];
}

static __forceinline__ __device__ void writeHit(int materialId, glm::vec3 normal, glm::vec2 uv, bool outside)
{
    const unsigned int i = optixGetLaunchIndex().x;

    ShadeableIntersection isect;
    isect.t = optixGetRayTmax();
    isect.surfaceNormal = normal;
    isect.uv = uv;
    isect.materialId = materialId;
    isect.outside = outside;
    params.intersections[i] = isect;
    params.materialIds[i] = materialId;
}

// Unit cube GAS: 12 triangles, two per face, in the order -x +x -y +y -z +z
// (see buildCubeGas), so the primitive index says which face was hit. Like
// boxIntersectionTest the normal faces the ray: the face's outward normal
// when the ray came from outside, the inward one when the ray started inside
// the cube and is leaving (the exit hit of a refractive cube).
extern "C" __global__ void __closesthit__cube()
{
    const unsigned int face = optixGetPrimitiveIndex() / 2;
    float n[3] = { 0.0f, 0.0f, 0.0f };
    n[face / 2] = (face & 1) ? 1.0f : -1.0f;

    // Object to world for a normal is the inverse transpose, which handles the
    // non-uniform scales the scene files use for walls.
    const float3 worldNormal = optixTransformNormalFromObjectToWorldSpace(make_float3(n[0], n[1], n[2]));
    // A closest-hit program cannot read the object-space ray (only
    // intersection and any-hit programs can), so the world ray is transformed
    // by hand with the hit instance's transform.
    const float3 origin = optixTransformPointFromWorldToObjectSpace(optixGetWorldRayOrigin());
    const bool outside = fmaxf(fabsf(origin.x), fmaxf(fabsf(origin.y), fabsf(origin.z))) > 0.5f;

    glm::vec3 normal = glm::normalize(toVec3(worldNormal));
    if (!outside)
    {
        normal = -normal;
    }
    writeHit(instance().materialId, normal, glm::vec2(0.0f), outside);
}

// Unit sphere GAS: one custom primitive, center at the origin, radius 0.5,
// intersected by the program below. OptiX's built-in sphere primitive is not
// usable here: it is hollow, back faces are culled, and a ray that starts
// inside never hits it (Programming Guide 9.1, "Curves and Spheres"), which
// is exactly the ray a refractive sphere sends out after the first bounce.
//
// Same quadratic and same inside rule as sphereIntersectionTest. The
// object-space direction is left unnormalized on purpose: OptiX derives it
// from the (normalized) world ray, so the same t parameterizes both spaces
// and the reported t is the world distance shadeMaterial needs.
extern "C" __global__ void __intersection__sphere()
{
    const glm::vec3 o = toVec3(optixGetObjectRayOrigin());
    const glm::vec3 d = toVec3(optixGetObjectRayDirection());

    // |o + t d|^2 = r^2  ->  a t^2 + 2 b t + c = 0
    const float a = glm::dot(d, d);
    const float b = glm::dot(o, d);
    const float c = glm::dot(o, o) - 0.25f;
    const float discriminant = b * b - a * c;
    if (discriminant < 0.0f)
    {
        return;
    }
    const float root = sqrtf(discriminant);
    const float tNear = (-b - root) / a;
    const float tFar = (-b + root) / a;

    // Both roots ahead: the ray came from outside and enters at the near one.
    // Only the far root ahead: the ray started inside and leaves through it.
    float t;
    bool outside;
    if (tNear > 0.0f && tFar > 0.0f)
    {
        t = tNear;
        outside = true;
    }
    else if (tFar > 0.0f)
    {
        t = tFar;
        outside = false;
    }
    else
    {
        return;
    }

    // Attributes for the closest-hit program: the object-space hit point,
    // which is also the normal direction for a sphere at the origin, and the
    // inside/outside decision, so it is made in one place.
    const glm::vec3 hit = o + t * d;
    optixReportIntersection(t, 0,
        __float_as_uint(hit.x), __float_as_uint(hit.y), __float_as_uint(hit.z),
        static_cast<unsigned int>(outside));
}

// Like sphereIntersectionTest the normal is flipped to face the ray when the
// ray started inside.
extern "C" __global__ void __closesthit__sphere()
{
    const float3 hit = make_float3(__uint_as_float(optixGetAttribute_0()),
                                   __uint_as_float(optixGetAttribute_1()),
                                   __uint_as_float(optixGetAttribute_2()));
    const bool outside = optixGetAttribute_3() != 0;

    glm::vec3 normal = glm::normalize(toVec3(optixTransformNormalFromObjectToWorldSpace(hit)));
    if (!outside)
    {
        normal = -normal;
    }

    writeHit(instance().materialId, normal, glm::vec2(0.0f), outside);
}

// Mesh GAS: OptiX's built-in triangle intersection reports which triangle of
// the instance's TriangleMesh was hit and where on it (barycentrics u, v
// weighting vertices 1 and 2). The vertex data comes from the same flat
// arrays the naive kernel reads. Same conventions as meshIntersectionTest:
// the geometric normal decides outside, the interpolated vertex normal is
// the shading normal, flipped to face the ray on a back-face hit, and the uv
// is interpolated with the same weights.
extern "C" __global__ void __closesthit__mesh()
{
    const InstanceRecord inst = instance();
    const TriangleMesh mesh = params.buffers.meshes[inst.meshId];
    const glm::ivec3 tri = params.buffers.indices[mesh.indexOffset + optixGetPrimitiveIndex()];
    const glm::vec3 p0 = params.buffers.positions[tri.x];
    const glm::vec3 p1 = params.buffers.positions[tri.y];
    const glm::vec3 p2 = params.buffers.positions[tri.z];

    // Object-space direction, transformed by hand since a closest-hit program
    // cannot read the object ray. Only its sign against the geometric normal
    // matters, and that is the same in object and world space.
    const glm::vec3 d = toVec3(optixTransformVectorFromWorldToObjectSpace(optixGetWorldRayDirection()));
    const glm::vec3 geometricNormal = glm::cross(p1 - p0, p2 - p0);
    const bool outside = glm::dot(geometricNormal, d) < 0.0f;

    const float2 bary = optixGetTriangleBarycentrics();
    const glm::vec3 n = (1.0f - bary.x - bary.y) * params.buffers.normals[tri.x]
                      + bary.x * params.buffers.normals[tri.y]
                      + bary.y * params.buffers.normals[tri.z];
    glm::vec3 normal = glm::normalize(toVec3(optixTransformNormalFromObjectToWorldSpace(toFloat3(n))));
    if (!outside)
    {
        normal = -normal;
    }
    // Same fallback as meshIntersectionTest: a smooth vertex normal near a
    // silhouette can point away from the ray even on a front-face hit, and
    // scatterRay needs one that faces it, so use the geometric normal there.
    if (glm::dot(normal, toVec3(optixGetWorldRayDirection())) > 0.0f)
    {
        normal = glm::normalize(toVec3(optixTransformNormalFromObjectToWorldSpace(toFloat3(geometricNormal))));
        if (!outside)
        {
            normal = -normal;
        }
    }

    const glm::vec2 uv = (1.0f - bary.x - bary.y) * params.buffers.uvs[tri.x]
                       + bary.x * params.buffers.uvs[tri.y]
                       + bary.y * params.buffers.uvs[tri.z];

    writeHit(inst.materialId, normal, uv, outside);
}
