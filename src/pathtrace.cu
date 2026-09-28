#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/random.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"
#include "wavefront_ops.h"
#include "optix_intersect.h"

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
#endif // ERRORCHECK
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1u << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
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
// two ping-pong pairs: compactPaths and sortMaterials swap
// them with the spare buffers in wavefront_ops.cu, so they change every bounce.
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
// The scene's flat mesh arrays (see Scene), read by the naive kernel and by
// OptiX. Null pointers when the scene has no meshes.
static MeshBuffers dev_mesh = {};

// Device copy of a host vector, or null when it is empty.
template <typename T>
static T* uploadVector(const std::vector<T>& v)
{
    if (v.empty())
    {
        return nullptr;
    }
    T* d = nullptr;
    cudaMalloc(&d, v.size() * sizeof(T));
    cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
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

static ToneMapMode toneMapMode = TONEMAP_AGX;
static float toneMapExposure = 1.f;
void setToneMap(ToneMapMode mode, float exposure)
{
    toneMapMode = mode;
    toneMapExposure = exposure;
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // TODO: initialize any extra device memeory you need
    cudaMalloc(&dev_materialIds, pixelcount * sizeof(int));
    // initialize spare buffers and CUB storage for compaction and material sort
    wavefrontInit(pixelcount);

    dev_mesh.positions = uploadVector(scene->positions);
    dev_mesh.normals = uploadVector(scene->normals);
    dev_mesh.indices = uploadVector(scene->indices);
    dev_mesh.meshes = uploadVector(scene->meshes);

    // OptiX intersection stage: context, acceleration structures, pipeline
    // and shader binding table, built once for this scene. If the driver has
    // no OptiX or any step fails, computeIntersections stays in use.
    optixReady = useOptix && optixIntersectInit(scene, dev_mesh, optixValidation);
    if (useOptix && !optixReady)
    {
        fprintf(stderr, "OptiX unavailable, using the naive intersection kernel\n");
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
    // TODO: clean up any extra device memory you created
    cudaFree(dev_materialIds);
    if (optixReady)
    {
        optixIntersectFree();  // before the mesh buffers it points into
        optixReady = false;
    }
    cudaFree(dev_mesh.positions);
    cudaFree(dev_mesh.normals);
    cudaFree(dev_mesh.indices);
    cudaFree(dev_mesh.meshes);
    dev_mesh = {};
    // Frees the spares. dev_paths / dev_intersections above may hold the
    // workspace's original buffers by now; the two sides still free each
    // allocation exactly once.
    wavefrontFree();

    checkCUDAError("pathtraceFree");
}

void pathtraceReset()
{
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));
    checkCUDAError("pathtraceReset");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
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
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * (float(x + jx) - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)(y + jy) - (float)cam.resolution.y * 0.5f)
        );

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
    int num_paths,
    PathSegment* pathSegments,
    int* materialIDs,
    int numMaterials,
    Geom* geoms,
    int geoms_size,
    MeshBuffers buffers,
    ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool closest_outside;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;
        bool tmp_outside;

        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_outside);
            }
            else  // MESH: every triangle of the instanced mesh
            {
                t = meshIntersectionTest(geom, buffers.meshes[geom.meshId], buffers, pathSegment.ray,
                    tmp_intersect, tmp_normal, tmp_outside);
            }

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
                closest_outside = tmp_outside;
            }
        }

        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
            // one past the last material id: misses sort last and the key
            // range stays small enough for a one-pass radix sort
            materialIDs[path_index] = numMaterials;
        }
        else
        {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            materialIDs[path_index] = geoms[hit_geom_index].materialid;
            intersections[path_index].surfaceNormal = normal;
            intersections[path_index].outside = closest_outside;
        }
    }
}

// The intersection stage behind one call: OptiX when it initialized, else
// the naive kernel above. Both fill dev_intersections and dev_materialIds the
// same way, so nothing downstream knows which one ran.
static void intersectScene(int numPaths, int numMaterials)
{
    if (optixReady)
    {
        optixIntersect(numPaths, dev_paths, dev_intersections, dev_materialIds, numMaterials);
        return;
    }
    const int blockSize1d = 128;
    dim3 numBlocks = (numPaths + blockSize1d - 1) / blockSize1d;
    computeIntersections<<<numBlocks, blockSize1d>>>(
        numPaths,
        dev_paths,
        dev_materialIds,
        numMaterials,
        dev_geoms,
        hst_scene->geoms.size(),
        dev_mesh,
        dev_intersections
    );
}

__global__ void shadeMaterial(
    int iter,
    int num_paths,
    int cur_depth,
    bool russianRoulette,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    glm::vec3* image)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        // Work on a register copy of the segment: one coalesced load here and
        // one store at the end, instead of global traffic on every field access
        // (scatterRay takes it by reference, so it would otherwise hit global
        // memory for every read and write inside).
        PathSegment seg = pathSegments[idx];
        if (seg.remainingBounces <= 0) return;
        ShadeableIntersection intersection = shadeableIntersections[idx];
        
        
        if (intersection.t > 0.0f) // if the intersection exists...
        {
            // Set up the RNG
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, seg.pixelIndex, seg.remainingBounces);
            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;
            // glm::vec3 materialColor = intersection.surfaceNormal;

            // If the material indicates that the object was a light
            if (material.emittance > 0.0f) {
                seg.color *= (materialColor * material.emittance);
                seg.remainingBounces = 0;
            }
            else {
                // ran out of bounces without hitting light
                if (seg.remainingBounces == 1) {
                    seg.color = glm::vec3(0.f);
                    seg.remainingBounces = 0;
                } else {    // still have bounces keep it up
                    // Russian Roulette
                    if (russianRoulette && cur_depth >= 3)
                    {
                        glm::vec3 c = seg.color;
                        // pick probability based on luminance
                        float p = glm::min(0.95f, 0.2126f * c.r + 0.7152f * c.g + 0.0722f * c.b);
                        thrust::uniform_real_distribution<float> u01(0,1);
                        float rand = u01(rng);

                        if (rand >= p) {    // terminate
                            seg.remainingBounces = 0;
                            seg.color = glm::vec3(0.f);
                        } else {   
                            // lucky survive, boost weighting
                            seg.color /= p;
                            
                        }
                    }
                    
                    
                    // keep bouncing, guard is so that terminated rays from roulette above doesn't continue into scatterRay
                    if (seg.remainingBounces > 0) {
                        scatterRay(seg,
                            seg.ray.origin + intersection.t * seg.ray.direction,
                            intersection.surfaceNormal, intersection.outside, material, rng);
                        --seg.remainingBounces;
                    }
                }
            }


            // If there was no intersection, color the ray black. could add Alpha channel if want to composite later
        }
        else {
            seg.color = glm::vec3(0.0f);
            seg.remainingBounces = 0;
        }

        // A path that terminated in any branch above adds its color to the
        // image now, because compaction drops it next. Each pixel has exactly
        // one path per iteration, so no two threads add to the same pixel.
        if (seg.remainingBounces <= 0) {
            image[seg.pixelIndex] += seg.color;
        }

        pathSegments[idx] = seg;
    }
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
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

void pathtrace(uchar4* pbo, int frame, int iter)
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

    // 1D block for path tracing
    const int blockSize1d = 128;

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    // TODO: perform one iteration of path tracing

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    int num_paths = pixelcount;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    while (!iterationComplete)
    {
        // dev_intersections is not cleared between bounces: both intersection
        // paths write every entry, with t = -1 on a miss.

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        intersectScene(num_paths, numMaterials);
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        depth++;

        // TODO:
        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.

        // Material sort: group paths by the material they hit so shadeMaterial warps are branch-uniform.
        // Swaps dev_paths and dev_intersections for the sorted copies.
        if (useMaterialSort)
        {
            sortMaterials(num_paths, materialKeyBits, dev_materialIds, dev_intersections, dev_paths);
        }

        shadeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            num_paths,
            depth,
            useRussianRoulette,
            dev_intersections,
            dev_paths,
            dev_materials,
            dev_image
        );
        // Stream compaction: keep only the live paths. Terminated ones have
        // already added their color to dev_image in shadeMaterial. Swaps
        // dev_paths for the compacted copy.
        num_paths = compactPaths(dev_paths, num_paths);

        iterationComplete = num_paths == 0 || depth >= traceDepth;

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    // No gather pass: every path added its color to dev_image in shadeMaterial
    // when it terminated.

    ///////////////////////////////////////////////////////////////////////////

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
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
    checkCUDAError("pathtraceDownloadImage");
}
