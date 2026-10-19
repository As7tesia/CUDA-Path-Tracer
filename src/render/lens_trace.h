#pragma once

// Tracing a ray through a lens prescription (scene/lens.h), surface by
// surface. Host and device: the generate kernel traces camera rays with it,
// lens.cpp works out a lens's focal length and pupil with it, and the host
// test checks both against the published numbers.
//
// Lens space is millimeters. z runs along the optical axis from the front
// vertex (z = 0) toward the film (z > 0); the scene is at z < 0. A surface's
// radius is positive when its center of curvature lies toward the film, so
// the surface bulges toward the scene.

#include "scene/lens.h"

#include <glm/glm.hpp>

#include <cmath>

// What the generate kernel needs of a lens: the table on the device and the
// numbers the host worked out from it (LensSystem::Paraxial). surfaces is
// null when the scene has no lens.
struct LensDevice
{
    const LensSurface* surfaces;
    int count;
    float filmDistance;     // front vertex to film, mm
    float entrancePupilZ;   // the entrance pupil's center, mm behind the front vertex: Camera::position sits here
    float rearZ;            // the rear surface's vertex, mm behind the front vertex
    float rearRadius;       // its aperture radius: camera rays are aimed at a point of this disk
    float pixelMm;          // film millimeters per pixel
    float mmToScene;        // scene units per millimeter, 0.001 * UNITS_PER_METER
    int* blocked;           // where the kernel counts the camera rays the lens blocked; null = do not count
};

// Where the ray o + t d (d unit length) meets surface s, whose vertex is at
// z. Writes the hit and the unit normal facing the ray. False when the ray
// misses the sphere or the hit lies behind the ray.
#ifdef __CUDACC__
__host__ __device__
#endif
inline bool intersectLensSurface(const LensSurface& s, float z, glm::vec3 o, glm::vec3 d, glm::vec3& p, glm::vec3& n)
{
    if (s.radius == 0.0f)
    {
        // Flat: the stop, or a plano face
        if (d.z == 0.0f)
        {
            return false;
        }
        const float t = (z - o.z) / d.z;
        if (t <= 0.0f)
        {
            return false;
        }
        p = o + t * d;
        n = glm::vec3(0.0f, 0.0f, d.z > 0.0f ? -1.0f : 1.0f);
        return true;
    }
    const float R = s.radius;
    const glm::vec3 c(0.0f, 0.0f, z + R);
    const glm::vec3 oc = o - c;
    // |oc + t d|^2 = R^2: t^2 + 2 b t + cc = 0
    const float b = glm::dot(oc, d);
    const float cc = glm::dot(oc, oc) - R * R;
    const float disc = b * b - cc;
    if (disc < 0.0f)
    {
        return false;
    }
    const float sq = sqrtf(disc);
    // The lens surface is the half of the sphere around the vertex. A ray
    // toward the film (d.z > 0) reaches it as the nearer root when the
    // surface bulges toward the scene (R > 0) and as the farther root when
    // it is hollow; toward the scene the two swap.
    const bool nearer = (d.z > 0.0f) == (R > 0.0f);
    const float t = nearer ? -b - sq : -b + sq;
    if (t <= 0.0f)
    {
        return false;
    }
    p = o + t * d;
    n = (p - c) / fabsf(R);
    if (glm::dot(n, d) > 0.0f)
    {
        n = -n;
    }
    return true;
}

// Snell's law across a surface with unit normal n facing the ray, from index
// n1 into n2. False at total internal reflection, where the camera trace
// treats the ray as blocked (internal reflections belong to the lens flare
// pass).
#ifdef __CUDACC__
__host__ __device__
#endif
inline bool refractLens(glm::vec3 d, glm::vec3 n, float n1, float n2, glm::vec3& out)
{
    const float eta = n1 / n2;
    const float cosI = -glm::dot(n, d);
    const float sin2T = eta * eta * (1.0f - cosI * cosI);
    if (sin2T >= 1.0f)
    {
        return false;
    }
    out = eta * d + (eta * cosI - sqrtf(1.0f - sin2T)) * n;
    return true;
}

// Traces the ray o, d (d unit length) through surfaces from index from to
// index to, one after the other: toward the film when to > from, toward the
// scene when to < from, one surface when they are equal. zFrom is the vertex
// of surface from. On return o and d are the ray as it leaves the last
// surface crossed. False when the lens blocks the ray: a hit outside a
// surface's clear aperture or the stop's opening, a missed surface, or total
// internal reflection. Aspheric terms are not applied yet: a surface is
// traced as the sphere of its paraxial radius.
#ifdef __CUDACC__
__host__ __device__
#endif
inline bool traceLens(const LensSurface* s, int from, int to, float zFrom, glm::vec3& o, glm::vec3& d)
{
    const int step = to >= from ? 1 : -1;
    float z = zFrom;
    for (int i = from;; i += step)
    {
        const LensSurface& surf = s[i];
        glm::vec3 p;
        glm::vec3 n;
        if (!intersectLensSurface(surf, z, o, d, p, n))
        {
            return false;
        }
        if (p.x * p.x + p.y * p.y > surf.apertureRadius * surf.apertureRadius)
        {
            return false;
        }
        o = p;
        if (!surf.isStop)
        {
            // The glass behind the surface is its own ior; the medium on its
            // scene side is the previous surface's, air before the first
            const float sceneSide = i > 0 ? s[i - 1].ior : 1.0f;
            const float n1 = step > 0 ? sceneSide : surf.ior;
            const float n2 = step > 0 ? surf.ior : sceneSide;
            if (n1 != n2 && !refractLens(d, n, n1, n2, d))
            {
                return false;
            }
        }
        if (i == to)
        {
            return true;
        }
        z += step > 0 ? surf.thickness : -s[i - 1].thickness;
    }
}
