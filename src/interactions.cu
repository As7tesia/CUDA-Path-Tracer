#include "interactions.h"

#include "utilities.h"

#include <thrust/random.h>

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    bool outside,
    const Material &m,
    thrust::default_random_engine &rng)
{
    // Every case must do two things:
    //   1. set pathSegment.ray (origin nudged off the surface, new direction)
    //   2. multiply pathSegment.color by f * cos(theta) / pdf for that direction
    const glm::vec3 in = pathSegment.ray.direction;

    switch (m.type)
    {
    case DIFFUSE:
    {
        // f = albedo/pi, pdf = cos/pi, so the weight is just albedo
        pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
        pathSegment.ray.origin = intersect + normal * EPSILON;
        pathSegment.color *= m.color;
        break;
    }
    case SPECULAR:
    {
        // perfect mirror just reflect the ray
        pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
        pathSegment.ray.origin = intersect + normal * EPSILON;
        pathSegment.color *= m.specular.color;
        break;
    }
    case REFRACTIVE:
    {
        //  - need to know entering vs leaving: normal always faces the incoming ray
        float eta = outside ? 1.f / m.ior : m.ior;
        float cosTheta = -glm::dot(in, normal);

        float r0 = (1.f - eta) / (1.f + eta);
        r0 = r0 * r0;
        float R = r0 + (1.f - r0) * powf(1.f - cosTheta, 5.f);

        thrust::uniform_real_distribution<float> u01(0, 1);
        glm::vec3 refracted = glm::refract(in, normal, eta);
        bool tir = glm::dot(refracted, refracted) < EPSILON;

        if (tir || u01(rng) < R)
        {
            pathSegment.ray.direction = glm::reflect(in, normal);
            pathSegment.ray.origin = intersect + normal * EPSILON;
        }
        else
        {
            pathSegment.ray.direction = refracted;
            pathSegment.ray.origin = intersect - normal * EPSILON;
        }
        pathSegment.color *= m.color;
        break;
    }
    default:
        // EMISSIVE is handled in the shade kernel before scatterRay is called
        break;
    }
}
