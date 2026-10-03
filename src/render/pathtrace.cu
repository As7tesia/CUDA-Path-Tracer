#include "render/pathtrace.h"

#include <cstdio>
#include <cmath>
#include <vector>
#include <thrust/random.h>

#include "scene/sceneStructs.h"
#include "scene/scene.h"
#include "glm/glm.hpp"
#include "timing.h"
#include "utilities.h"
#include "render/intersections.h"
#include "render/mesh_hit.h"
#include "render/bsdf.h"
#include "render/pbr_surface.h"
#include "render/sampling.h"
#include "render/wavefront_ops.h"
#include "optix/optix_intersect.h"
#include "render/textures.h"

// After kernel launches: waits for the device and ends the program, naming
// msg, if anything launched since the last check failed (the error a kernel
// hits only shows once the device has run it). ERRORCHECK 0 skips both, and
// the explicit wait after the intersection stage. Off in Release since the
// 2026-10-02 profile: the two waits per bounce cost 5 to 10% of a sample at
// 1024x1024 and up to 20% at 400x400 (README, Performance). A kernel error
// still ends the run, at wavefrontSwapPaths' stream wait in the same bounce,
// with that call's name instead of the stage's. -DERRORCHECK=1 turns it back
// on.
#ifndef ERRORCHECK
#ifdef NDEBUG
#define ERRORCHECK 0
#else
#define ERRORCHECK 1
#endif
#endif

#define checkCUDAError(msg) checkCUDAErrorFn(msg, __FILE__, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaCheck(cudaGetLastError(), msg, file, line);
#endif // ERRORCHECK
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image,
    ToneMapMode toneMap, float exposure)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index] / (float)iter;   // running average, scene-linear
        pix = applyToneMap(pix, toneMap, exposure);   // display-encoded [0, 1]

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static int* dev_materialIds = NULL;
// two ping-pong pairs: wavefrontSwapPaths (after shading) and sortMaterials
// swap them with the spare buffers in wavefront_ops.cu, so they change every
// bounce.
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
// The scene's flat mesh arrays (see Scene), read by the naive kernel and by
// OptiX. Null pointers when the scene has no meshes.
static MeshBuffers dev_mesh = {};
// Texture objects indexed like Scene::textures (textures.cpp), null when the
// scene has none.
static cudaTextureObject_t* dev_textures = NULL;

// Device copy of a host vector, or null when it is empty.
template <typename T>
static T* uploadVector(const std::vector<T>& v)
{
    if (v.empty())
    {
        return nullptr;
    }
    T* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, v.size() * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

void setGuiData(GuiDataContainer* data)
{
    guiData = data;
}

static bool useRussianRoulette = true;
void setRussianRoulette(bool enabled) { useRussianRoulette = enabled; }

static bool useMaterialSort = true;
void setMaterialSort(bool enabled) { useMaterialSort = enabled; }

// OptiX intersection stage. useOptix is the request; optixReady is whether
// optixIntersectInit succeeded, which is what intersectScene checks.
static bool useOptix = true;
static bool optixValidation = false;
static bool optixReady = false;
void setOptix(bool enabled) { useOptix = enabled; }
void setOptixValidation(bool enabled) { optixValidation = enabled; }

static ToneMapMode toneMapMode = TONEMAP_AGX_PUNCHY;
static float toneMapExposure = 1.f;
void setToneMap(ToneMapMode mode, float exposure)
{
    toneMapMode = mode;
    toneMapExposure = exposure;
}

// --timing (timing.h): cudaEvents around every stage of every bounce. The
// time between two consecutive events is what the GPU timeline shows between
// them, so a stage's time is its kernels plus the launch gaps and any host
// synchronization inside it (the error checks, wavefrontSwapPaths' readback
// of the alive count, which is all the compact stage holds since compaction
// moved into shadeMaterial). Sums over iterations; pathtraceTimingReport
// averages.
namespace
{
enum Stage { STAGE_INTERSECT, STAGE_SORT, STAGE_SHADE, STAGE_COMPACT, NUM_STAGES };
const char* const STAGE_NAMES[NUM_STAGES] = { "intersect", "sort", "shade", "compact" };

struct StageTiming
{
    int maxDepth = 0;                 // bounces the events cover
    int iterations = 0;               // iterations summed so far
    double generateMs = 0.0;
    std::vector<double> stageMs;      // [depth * NUM_STAGES + stage]
    std::vector<double> alive;        // [depth]: paths entering bounce depth; [maxDepth]: left after the last
    std::vector<cudaEvent_t> events;  // 0 before generate, 1 after, then 2 + depth * NUM_STAGES + stage

    int eventIndex(int depth, int stage) const { return 2 + depth * NUM_STAGES + stage; }
    bool ready(int traceDepth) const { return !events.empty() && traceDepth <= maxDepth; }
} stageTiming;

void stageTimingFree()
{
    for (cudaEvent_t e : stageTiming.events)
    {
        cudaEventDestroy(e);
    }
    stageTiming = StageTiming();
}

void stageTimingInit(int maxDepth)
{
    stageTimingFree();
    stageTiming.maxDepth = maxDepth;
    stageTiming.stageMs.assign((size_t)maxDepth * NUM_STAGES, 0.0);
    stageTiming.alive.assign((size_t)maxDepth + 1, 0.0);
    stageTiming.events.resize(2 + (size_t)maxDepth * NUM_STAGES);
    for (cudaEvent_t& e : stageTiming.events)
    {
        CUDA_CHECK(cudaEventCreate(&e));
    }
}

// Waits for the iteration's last event and adds every interval to the sums.
void stageTimingAccumulate(int bounces)
{
    StageTiming& st = stageTiming;
    CUDA_CHECK(cudaEventSynchronize(st.events[st.eventIndex(bounces - 1, STAGE_COMPACT)]));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, st.events[0], st.events[1]));
    st.generateMs += ms;
    int previous = 1;
    for (int depth = 0; depth < bounces; ++depth)
    {
        for (int stage = 0; stage < NUM_STAGES; ++stage)
        {
            const int current = st.eventIndex(depth, stage);
            CUDA_CHECK(cudaEventElapsedTime(&ms, st.events[previous], st.events[current]));
            st.stageMs[(size_t)depth * NUM_STAGES + stage] += ms;
            previous = current;
        }
    }
    ++st.iterations;
}
}  // namespace

void pathtraceTimingReport()
{
    const StageTiming& st = stageTiming;
    if (!timingEnabled() || st.iterations == 0)
    {
        return;
    }
    printf("TIMING,size.PathSegment_bytes,%zu\n", sizeof(PathSegment));
    printf("TIMING,size.ShadeableIntersection_bytes,%zu\n", sizeof(ShadeableIntersection));
    printf("TIMING,render.generate,%.4f\n", st.generateMs / st.iterations);
    printf("BOUNCE,depth,alive_in,alive_out");
    for (int stage = 0; stage < NUM_STAGES; ++stage)
    {
        printf(",%s", STAGE_NAMES[stage]);
    }
    printf("\n");
    for (int depth = 0; depth < st.maxDepth; ++depth)
    {
        printf("BOUNCE,%d,%.1f,%.1f", depth + 1, st.alive[depth] / st.iterations, st.alive[depth + 1] / st.iterations);
        for (int stage = 0; stage < NUM_STAGES; ++stage)
        {
            printf(",%.4f", st.stageMs[(size_t)depth * NUM_STAGES + stage] / st.iterations);
        }
        printf("\n");
    }
}

void pathtraceInit(Scene* scene)
{
    TimingScope timing("init.total");
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // Device memory before and after, for the scene's footprint
    size_t freeBefore = 0;
    size_t totalMemory = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeBefore, &totalMemory));

    CUDA_CHECK(cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3)));
    CUDA_CHECK(cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3)));

    CUDA_CHECK(cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment)));

    dev_geoms = uploadVector(scene->geoms);
    dev_materials = uploadVector(scene->materials);

    CUDA_CHECK(cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection)));
    CUDA_CHECK(cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection)));

    // The material sort's key per path, written by the intersection stage
    CUDA_CHECK(cudaMalloc(&dev_materialIds, pixelcount * sizeof(int)));
    // Spare buffers and CUB storage for compaction and the material sort
    wavefrontInit(pixelcount);

    {
        TimingScope upload("init.mesh_upload");
        dev_mesh.positions = uploadVector(scene->positions);
        dev_mesh.normals = uploadVector(scene->normals);
        dev_mesh.uvs = uploadVector(scene->uvs);
        dev_mesh.tangents = uploadVector(scene->tangents);
        dev_mesh.indices = uploadVector(scene->indices);
        dev_mesh.meshes = uploadVector(scene->meshes);
    }

    {
        TimingScope upload("init.texture_upload");
        dev_textures = texturesInit(*scene);
    }

    // OptiX intersection stage: context, acceleration structures, pipeline
    // and shader binding table, built once for this scene. If the driver has
    // no OptiX or any step fails, computeIntersections stays in use.
    optixReady = useOptix && optixIntersectInit(scene, dev_mesh, dev_materials, dev_textures, optixValidation);
    if (useOptix && !optixReady)
    {
        fprintf(stderr, "OptiX unavailable, using the naive intersection kernel\n");
    }

    if (timingEnabled())
    {
        stageTimingInit(hst_scene->state.traceDepth);
        size_t freeAfter = 0;
        CUDA_CHECK(cudaMemGetInfo(&freeAfter, &totalMemory));
        timingAdd("init.gpu_footprint_mb", (double)(freeBefore - freeAfter) / (1024.0 * 1024.0));
    }

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    cudaFree(dev_materialIds);
    if (optixReady)
    {
        optixIntersectFree();  // before the mesh buffers it points into
        optixReady = false;
    }
    cudaFree(dev_mesh.positions);
    cudaFree(dev_mesh.normals);
    cudaFree(dev_mesh.uvs);
    cudaFree(dev_mesh.tangents);
    cudaFree(dev_mesh.indices);
    cudaFree(dev_mesh.meshes);
    dev_mesh = {};
    texturesFree();
    dev_textures = NULL;
    // Frees the spares. dev_paths / dev_intersections above may hold the
    // workspace's original buffers by now; the two sides still free each
    // allocation exactly once.
    wavefrontFree();
    stageTimingFree();

    checkCUDAError("pathtraceFree");
}

void pathtraceReset()
{
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;
    CUDA_CHECK(cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3)));
}

// One path per pixel, starting at the camera: a ray through a random point
// in the pixel (antialiasing), white throughput and traceDepth bounces left.
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        // jitter the x and y pixel coordinates as floats
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::uniform_real_distribution<float> u01(0, 1);
        float jx = u01(rng);   // in [0, 1)
        float jy = u01(rng);
        // Pixel (0, 0) is the top left of the image, as the PNG and the
        // viewport texture store it: x runs along right, y against up.
        segment.ray.direction = glm::normalize(cam.view
            + cam.right * cam.pixelLength.x * (float(x + jx) - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)(y + jy) - (float)cam.resolution.y * 0.5f)
        );

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
        segment.medium = -1;  // the camera sits in air
    }
}

// The naive intersection stage: every path tests every Geom (every triangle
// of a mesh), closest hit wins. Writes the hit, or t = -1 on a miss, and the
// material sort key; the shade kernel makes the next ray. iter seeds the
// ALPHA_BLEND test.
__global__ void computeIntersections(
    int iter,
    int numPaths,
    PathSegment* pathSegments,
    int* materialIds,
    int numMaterials,
    Geom* geoms,
    int numGeoms,
    MeshBuffers buffers,
    const Material* materials,
    const cudaTextureObject_t* textures,
    ShadeableIntersection* intersections)
{
    int pathIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (pathIndex < numPaths)
    {
        PathSegment pathSegment = pathSegments[pathIndex];
        const unsigned int alphaSeed = alphaPathSeed(iter, pathSegment.pixelIndex, pathSegment.remainingBounces);

        float t;
        glm::vec3 normal;
        glm::vec2 uv;
        glm::vec4 tangent;
        float tMin = FLT_MAX;
        int hitGeomIndex = -1;
        bool closestOutside;

        glm::vec3 tmpPoint;
        glm::vec3 tmpNormal;
        glm::vec2 tmpUv;
        glm::vec4 tmpTangent;
        bool tmpOutside;

        for (int i = 0; i < numGeoms; i++)
        {
            Geom& geom = geoms[i];

            // Only meshes have texture coordinates and tangents.
            tmpUv = glm::vec2(0.0f);
            tmpTangent = glm::vec4(0.0f);
            if (geom.type == GeomType::CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmpPoint, tmpNormal, tmpOutside);
            }
            else if (geom.type == GeomType::SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmpPoint, tmpNormal, tmpOutside);
            }
            else  // MESH: every triangle of the instanced mesh
            {
                t = meshIntersectionTest(geom, i, buffers.meshes[geom.meshId], buffers, materials[geom.materialId],
                    textures, alphaSeed, pathSegment.ray, tmpPoint, tmpNormal, tmpUv, tmpTangent, tmpOutside);
            }

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && tMin > t)
            {
                tMin = t;
                hitGeomIndex = i;
                normal = tmpNormal;
                uv = tmpUv;
                tangent = tmpTangent;
                closestOutside = tmpOutside;
            }
        }

        if (hitGeomIndex == -1)
        {
            intersections[pathIndex].t = -1.0f;
            // one past the last material id: misses sort last and the key
            // range stays small enough for a one-pass radix sort
            materialIds[pathIndex] = numMaterials;
        }
        else
        {
            // The ray hits something
            intersections[pathIndex].t = tMin;
            intersections[pathIndex].materialId = geoms[hitGeomIndex].materialId;
            materialIds[pathIndex] = geoms[hitGeomIndex].materialId;
            intersections[pathIndex].surfaceNormal = normal;
            intersections[pathIndex].uv = uv;
            intersections[pathIndex].tangent = tangent;
            intersections[pathIndex].outside = closestOutside;
        }
    }
}

// The intersection stage behind one call: OptiX when it initialized, else
// the naive kernel above. Both fill dev_intersections and dev_materialIds the
// same way, so nothing downstream knows which one ran.
static void intersectScene(int iter, int numPaths, int numMaterials)
{
    if (optixReady)
    {
        optixIntersect(iter, numPaths, dev_paths, dev_intersections, dev_materialIds, numMaterials);
        return;
    }
    dim3 numBlocks = (numPaths + PATH_BLOCK_SIZE - 1) / PATH_BLOCK_SIZE;
    computeIntersections<<<numBlocks, PATH_BLOCK_SIZE>>>(
        iter,
        numPaths,
        dev_paths,
        dev_materialIds,
        numMaterials,
        dev_geoms,
        hst_scene->geoms.size(),
        dev_mesh,
        dev_materials,
        dev_textures,
        dev_intersections
    );
}

// The survivor append below takes whole warps.
static_assert(PATH_BLOCK_SIZE % 32 == 0, "shadeMaterial's blocks must be whole warps");

// Shades paths[0, numPaths) and appends the paths that go on to
// survivors.paths, compacted: the compaction is fused into shading, so no
// separate pass reads the paths again to drop the ones that ended.
__global__ void shadeMaterial(
    int iter,
    int numPaths,
    int depth,
    bool russianRoulette,
    ShadeableIntersection* shadeableIntersections,
    const PathSegment* pathSegments,
    SurvivorBuffer survivors,
    Material* materials,
    const cudaTextureObject_t* textures,
    glm::vec3* image)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Every lane of the warp has to reach the ballot at the end, so a lane
    // with nothing to shade (past numPaths, or a path already ended) skips the
    // shading instead of returning.
    //
    // Work on a register copy of the segment: one coalesced load here and one
    // store at the end, instead of global traffic on every field access
    // (scatterPbr takes it by reference, so it would otherwise hit global
    // memory for every read and write inside).
    PathSegment seg;
    bool alive = false;
    if (idx < numPaths)
    {
        seg = pathSegments[idx];
        alive = seg.remainingBounces > 0;
    }
    if (alive)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];

        // A path ends by setting remainingBounces to 0 and is then left out
        // of the survivors. Its color is not read again: light reaches the
        // image only through the emission added below.
        if (intersection.t <= 0.0f)
        {
            // A miss. Nothing lights the scene from outside yet.
            seg.remainingBounces = 0;
        }
        else
        {
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, seg.pixelIndex, seg.remainingBounces);
            const Material& material = materials[intersection.materialId];
            const glm::vec3 hitPoint = seg.ray.origin + intersection.t * seg.ray.direction;
            const glm::vec3 wo = -seg.ray.direction;

            // Inside a volume (the path refracted into a material and has not
            // refracted out), the segment that just ended was absorbed along
            // its length t (Beer-Lambert), whatever it ended on.
            if (seg.medium >= 0)
            {
                const glm::vec3 absorption = materials[seg.medium].absorption;
                if (absorption != glm::vec3(0.0f))
                {
                    seg.color *= glm::exp(-absorption * intersection.t);
                }
            }

            // The texture lookups happen once per hit. On the last bounce the
            // path ends here, so only the emission is needed.
            const bool lastBounce = seg.remainingBounces == 1;
            PbrSurface surface;
            glm::vec3 emission;
            if (lastBounce)
            {
                emission = pbrEmission(material, intersection, wo, textures);
            }
            else
            {
                surface = pbrSurface(material, intersection, wo, textures);
                emission = surface.emission;
            }
            // An emitting surface adds its light and the path goes on: an
            // emitter also reflects, like any other surface. Each pixel has
            // exactly one path per iteration, so no two threads add to the
            // same pixel.
            if (emission != glm::vec3(0.0f))
            {
                image[seg.pixelIndex] += seg.color * emission;
            }

            if (lastBounce)
            {
                seg.remainingBounces = 0;  // the next ray would not be traced
            }
            else
            {
                // Russian roulette from the third bounce: the path survives
                // with a probability that follows the luminance of its
                // throughput (at most 0.95), and a survivor's throughput is
                // divided by that probability, which keeps the estimate
                // unbiased.
                if (russianRoulette && depth >= 3)
                {
                    const glm::vec3 c = seg.color;
                    const float p = glm::min(0.95f, 0.2126f * c.r + 0.7152f * c.g + 0.0722f * c.b);
                    thrust::uniform_real_distribution<float> u01(0, 1);
                    if (u01(rng) >= p)
                    {
                        seg.remainingBounces = 0;
                    }
                    else
                    {
                        seg.color /= p;
                    }
                }

                if (seg.remainingBounces > 0)
                {
                    // A sample that carries no light ends the path.
                    if (scatterPbr(seg, hitPoint, intersection.surfaceNormal, intersection.outside, material, surface,
                            rng))
                    {
                        --seg.remainingBounces;
                        // A direction past the surface refracted into the
                        // object or out of it: the medium for the absorption
                        // above travels with the path.
                        if (glm::dot(seg.ray.direction, intersection.surfaceNormal) < 0.0f)
                        {
                            seg.medium = intersection.outside ? intersection.materialId : -1;
                        }
                    }
                    else
                    {
                        seg.remainingBounces = 0;
                    }
                }
            }
        }

        alive = seg.remainingBounces > 0;
    }

    // The warp's survivors take consecutive slots of survivors.paths: lane 0
    // reserves them with one atomicAdd for the whole warp, and each survivor
    // writes to the reserved base plus the number of survivors in the lanes
    // below it. A path that ended is not written anywhere.
    const unsigned int warpSurvivors = __ballot_sync(0xffffffffu, alive);
    const unsigned int lane = threadIdx.x % 32;
    int base = 0;
    if (lane == 0 && warpSurvivors != 0)
    {
        base = atomicAdd(survivors.count, __popc(warpSurvivors));
    }
    base = __shfl_sync(0xffffffffu, base, 0);
    if (alive)
    {
        const unsigned int lanesBelow = (1u << lane) - 1u;
        survivors.paths[base + __popc(warpSurvivors & lanesBelow)] = seg;
    }

    // The next bounce appends into nextCount. It last held the survivors of
    // the bounce two back, and nothing reads it anymore.
    if (idx == 0)
    {
        *survivors.nextCount = 0;
    }
}

// Number of bits needed to hold every value in [0, maxValue]. Sets the radix
// sort's key range so it runs one pass for a scene with few materials.
static int bitsToHold(int maxValue)
{
    int bits = 1;
    while ((1 << bits) <= maxValue)
    {
        ++bits;
    }
    return bits;
}

void pathtrace(uchar4* pbo, int iter)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;
    const int numMaterials = (int)hst_scene->materials.size();
    // Sort keys lie in [0, numMaterials]; numMaterials is the miss key.
    const int materialKeyBits = bitsToHold(numMaterials);

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // The timing events record on the default stream, like every launch here
    const bool timing = timingEnabled() && stageTiming.ready(traceDepth);
    if (timing)
    {
        CUDA_CHECK(cudaEventRecord(stageTiming.events[0]));
    }
    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");
    if (timing)
    {
        CUDA_CHECK(cudaEventRecord(stageTiming.events[1]));
    }

    int depth = 0;
    int numPaths = pixelcount;

    // One bounce per pass: intersect, sort by material, and shade, which makes
    // the next rays and compacts away the paths that ended.
    bool iterationComplete = false;
    while (!iterationComplete)
    {
        // dev_intersections is not cleared between bounces: both intersection
        // paths write every entry, with t = -1 on a miss.
        if (timing)
        {
            stageTiming.alive[depth] += numPaths;
        }

        // tracing
        dim3 numBlocks = (numPaths + PATH_BLOCK_SIZE - 1) / PATH_BLOCK_SIZE;
        intersectScene(iter, numPaths, numMaterials);
        checkCUDAError("trace one bounce");
#if ERRORCHECK
        cudaDeviceSynchronize();
#endif
        if (timing)
        {
            CUDA_CHECK(cudaEventRecord(stageTiming.events[stageTiming.eventIndex(depth, STAGE_INTERSECT)]));
        }

        // Material sort: group paths by the material they hit, so the threads
        // of a shadeMaterial warp read the same material and mostly pick the
        // same lobes. Swaps dev_paths and dev_intersections for the sorted
        // copies.
        if (useMaterialSort)
        {
            sortMaterials(numPaths, materialKeyBits, dev_materialIds, dev_intersections, dev_paths);
        }
        if (timing)
        {
            CUDA_CHECK(cudaEventRecord(stageTiming.events[stageTiming.eventIndex(depth, STAGE_SORT)]));
        }

        // Shading writes the paths that go on, compacted, into the survivor
        // buffer; the light of the others is already in the image.
        shadeMaterial<<<numBlocks, PATH_BLOCK_SIZE>>>(
            iter,
            numPaths,
            depth + 1,
            useRussianRoulette,
            dev_intersections,
            dev_paths,
            wavefrontSurvivors(),
            dev_materials,
            dev_textures,
            dev_image
        );
        if (timing)
        {
            CUDA_CHECK(cudaEventRecord(stageTiming.events[stageTiming.eventIndex(depth, STAGE_SHADE)]));
        }
        // Swaps dev_paths for the survivors and reads their count back.
        numPaths = wavefrontSwapPaths(dev_paths);
        if (timing)
        {
            CUDA_CHECK(cudaEventRecord(stageTiming.events[stageTiming.eventIndex(depth, STAGE_COMPACT)]));
        }
        depth++;

        iterationComplete = numPaths == 0 || depth >= traceDepth;

        if (guiData != NULL)
        {
            guiData->tracedDepth = depth;
        }
    }
    if (timing)
    {
        stageTiming.alive[depth] += numPaths;  // survivors of the last bounce run
        stageTimingAccumulate(depth);
    }

    // No gather pass: shadeMaterial adds light to dev_image at the hit that
    // emits it.

    // Send results to OpenGL buffer for rendering (pbo is null in headless mode)
    if (pbo != nullptr)
    {
        sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image,
            toneMapMode, toneMapExposure);
    }

    checkCUDAError("pathtrace");
}

void pathtraceDownloadImage()
{
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;
    CUDA_CHECK(cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost));
}
