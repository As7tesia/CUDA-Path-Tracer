#include "intersections.h"

#include <cfloat>

__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/
        {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n(0.0f);
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin)
            {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax)
            {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0)
    {
        outside = true;
        if (tmin <= 0)
        {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }

    return -1;
}

__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0)
    {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0)
    {
        return -1;
    }
    else if (t1 > 0 && t2 > 0)
    {
        t = min(t1, t2);
        outside = true;
    }
    else
    {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside)
    {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}

// Moller-Trumbore, both faces. Solves o + t d = p0 + u (p1 - p0) + v (p2 - p0)
// for t, u, v and accepts the hit if it is ahead of the origin and inside the
// triangle. u and v are the barycentric weights of p1 and p2, so the caller
// can interpolate vertex data with (1 - u - v, u, v).
__host__ __device__ static bool rayTriangle(
    glm::vec3 o, glm::vec3 d,
    glm::vec3 p0, glm::vec3 p1, glm::vec3 p2,
    float& t, float& u, float& v)
{
    const glm::vec3 e1 = p1 - p0;
    const glm::vec3 e2 = p2 - p0;
    const glm::vec3 pvec = glm::cross(d, e2);
    const float det = glm::dot(e1, pvec);
    // det is (twice the area) * cos of the angle to the ray; zero means the ray
    // lies in the triangle's plane. Keeping the sign accepts back faces too.
    if (fabsf(det) < 1e-12f)
    {
        return false;
    }
    const float invDet = 1.0f / det;
    const glm::vec3 tvec = o - p0;
    u = glm::dot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f)
    {
        return false;
    }
    const glm::vec3 qvec = glm::cross(tvec, e1);
    v = glm::dot(d, qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f)
    {
        return false;
    }
    t = glm::dot(e2, qvec) * invDet;
    return t > 0.0f;
}

__host__ __device__ float meshIntersectionTest(
    const Geom& geom,
    const TriangleMesh& mesh,
    const MeshBuffers& buffers,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside)
{
    // Object space, with the direction normalized like the other tests, so
    // the object-space t is a distance there and the world distance comes
    // from the transformed hit point at the end.
    const glm::vec3 o = multiplyMV(geom.inverseTransform, glm::vec4(r.origin, 1.0f));
    const glm::vec3 d = glm::normalize(multiplyMV(geom.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tMin = FLT_MAX;
    int hit = -1;
    float hitU = 0.0f;
    float hitV = 0.0f;
    for (int i = 0; i < mesh.triCount; ++i)
    {
        const glm::ivec3 tri = buffers.indices[mesh.indexOffset + i];
        float t;
        float u;
        float v;
        if (rayTriangle(o, d, buffers.positions[tri.x], buffers.positions[tri.y], buffers.positions[tri.z], t, u, v)
            && t < tMin)
        {
            tMin = t;
            hit = i;
            hitU = u;
            hitV = v;
        }
    }
    if (hit < 0)
    {
        return -1;
    }

    const glm::ivec3 tri = buffers.indices[mesh.indexOffset + hit];
    const glm::vec3 p0 = buffers.positions[tri.x];
    const glm::vec3 p1 = buffers.positions[tri.y];
    const glm::vec3 p2 = buffers.positions[tri.z];

    // The geometric normal (glTF winds front faces counterclockwise) says
    // which side the ray came from. Its dot product with the direction has
    // the same sign in object and world space, so no transform is needed.
    const glm::vec3 geometricNormal = glm::cross(p1 - p0, p2 - p0);
    outside = glm::dot(geometricNormal, d) < 0.0f;

    // Shading normal: vertex normals weighted by the barycentrics, taken to
    // world space with the inverse transpose like the other tests.
    const glm::vec3 n = (1.0f - hitU - hitV) * buffers.normals[tri.x]
                      + hitU * buffers.normals[tri.y]
                      + hitV * buffers.normals[tri.z];
    normal = glm::normalize(multiplyMV(geom.invTranspose, glm::vec4(n, 0.0f)));
    if (!outside)
    {
        normal = -normal;
    }
    // Near a silhouette a smooth vertex normal can lean past the surface and
    // point away from the ray even on a front-face hit. scatterRay needs a
    // normal facing the ray, so those hits fall back to the geometric normal.
    if (glm::dot(normal, r.direction) > 0.0f)
    {
        normal = glm::normalize(multiplyMV(geom.invTranspose, glm::vec4(geometricNormal, 0.0f)));
        if (!outside)
        {
            normal = -normal;
        }
    }

    intersectionPoint = multiplyMV(geom.transform, glm::vec4(o + tMin * d, 1.0f));
    return glm::length(r.origin - intersectionPoint);
}
