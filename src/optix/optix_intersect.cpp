// OptiX intersection stage, host side. See optix_intersect.h.
//
// A .cpp on purpose: optix.h switches to the device API whenever __CUDACC__
// is defined, so host code that includes it cannot live in a .cu. The CUDA
// runtime calls below only need cuda_runtime.h.

#define NOMINMAX  // optix_stubs.h includes windows.h, whose min/max macros break glm

#include "optix/optix_intersect.h"

#include "optix/optix_params.h"
#include "scene/scene.h"
#include "timing.h"
#include "utilities.h"

#include <optix.h>
#include <optix_function_table_definition.h>  // exactly one translation unit defines the table
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

// The device programs from optix_programs.cu, compiled to OptiX-IR by nvcc
// and turned into this array by cmake/EmbedFile.cmake (see CMakeLists.txt).
extern const unsigned char optix_programs_ir[];
extern const size_t optix_programs_ir_size;

namespace
{
// Everything OptiX owns for the lifetime of the scene, created by
// optixIntersectInit and released by optixIntersectFree.
constexpr int NUM_GROUPS = 6;         // raygen, path miss, shadow miss, cube hit, sphere hit, mesh hit
constexpr int NUM_MISS_GROUPS = 2;    // miss index 0 = path rays, 1 = shadow rays
constexpr int NUM_HIT_GROUPS = 3;     // sbtOffset 0 = cube, 1 = sphere, 2 = mesh

OptixDeviceContext context = nullptr;
OptixModule module = nullptr;        // every program: raygen, miss, sphere intersection, closest hits
OptixProgramGroup groups[NUM_GROUPS] = {};
OptixPipeline pipeline = nullptr;
OptixShaderBindingTable sbt = {};
CUdeviceptr dev_sbtRecords = 0;
CUdeviceptr dev_cubeGas = 0;
CUdeviceptr dev_sphereGas = 0;
std::vector<CUdeviceptr> dev_meshGas;  // one GAS per TriangleMesh
CUdeviceptr dev_ias = 0;
OptixTraversableHandle iasHandle = 0;
OptixIntersectParams* dev_params = nullptr;
InstanceRecord* dev_instances = nullptr;  // instance id -> material and mesh
// pathtrace.cu's device arrays, passed through to the hit programs
MeshBuffers meshBuffers = {};
const Material* dev_materials = nullptr;
const cudaTextureObject_t* dev_textures = nullptr;

// Whether a Geom's hits go through the any-hit alpha test.
bool isAlphaTested(const Scene* scene, const Geom& g)
{
    return g.type == GeomType::MESH && scene->materials[g.materialId].alphaMode != ALPHA_OPAQUE;
}

bool check(OptixResult result, const char* call)
{
    if (result == OPTIX_SUCCESS)
    {
        return true;
    }
    fprintf(stderr, "OptiX: %s failed: %s (%s)\n", call,
        optixGetErrorName(result), optixGetErrorString(result));
    return false;
}

// Init stops at the first failure so pathtraceInit can fall back.
#define OPTIX_TRY(call) do { if (!check((call), #call)) return false; } while (0)

// Compile log from module, program group or pipeline creation, printed only
// when the call failed: on success it holds statistics, and warnings arrive
// through logCallback anyway.
void printLog(const char* what, OptixResult result, const char* log, size_t logSize)
{
    if (result != OPTIX_SUCCESS && logSize > 1)
    {
        fprintf(stderr, "OptiX %s log:\n%s\n", what, log);
    }
}

void logCallback(unsigned int level, const char* tag, const char* message, void*)
{
    fprintf(stderr, "OptiX [%u][%s]: %s\n", level, tag, message);
}

CUdeviceptr upload(const void* data, size_t bytes)
{
    void* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, bytes));
    CUDA_CHECK(cudaMemcpy(d, data, bytes, cudaMemcpyHostToDevice));
    return reinterpret_cast<CUdeviceptr>(d);
}

void release(CUdeviceptr& d)
{
    cudaFree(reinterpret_cast<void*>(d));
    d = 0;
}

// Builds one acceleration structure (GAS or IAS) from a single build input
// into a fresh output buffer, which stays allocated until optixIntersectFree.
bool buildAccel(const OptixBuildInput& input, CUdeviceptr& output, OptixTraversableHandle& handle)
{
    OptixAccelBuildOptions options = {};
    options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;  // trace speed over build speed (PREFER_FAST_BUILD is the opposite)
    options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes sizes = {};
    OPTIX_TRY(optixAccelComputeMemoryUsage(context, &options, &input, 1, &sizes));

    CUdeviceptr temp = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&temp), sizes.tempSizeInBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&output), sizes.outputSizeInBytes));
    OptixResult result = optixAccelBuild(context, 0, &options, &input, 1,
        temp, sizes.tempSizeInBytes, output, sizes.outputSizeInBytes, &handle, nullptr, 0);
    release(temp);  // cudaFree waits for the build, which runs on stream 0
    return check(result, "optixAccelBuild");
}

// Unit cube, corners at +-0.5: 8 vertices, 12 triangles, two per face in the
// order -x +x -y +y -z +z, wound counterclockwise seen from outside.
// __closesthit__cube derives the face normal from the primitive index, so the
// face order is part of the contract.
bool buildCubeGas(OptixTraversableHandle& handle)
{
    float3 vertices[8];
    for (int i = 0; i < 8; ++i)
    {
        vertices[i] = make_float3((i & 1) ? 0.5f : -0.5f,
                                  (i & 2) ? 0.5f : -0.5f,
                                  (i & 4) ? 0.5f : -0.5f);
    }
    const uint3 triangles[12] = {
        {0, 4, 6}, {0, 6, 2},   // -x
        {1, 7, 5}, {1, 3, 7},   // +x
        {0, 5, 4}, {0, 1, 5},   // -y
        {2, 6, 7}, {2, 7, 3},   // +y
        {0, 2, 3}, {0, 3, 1},   // -z
        {4, 7, 6}, {4, 5, 7},   // +z
    };
    CUdeviceptr dev_vertices = upload(vertices, sizeof(vertices));
    CUdeviceptr dev_triangles = upload(triangles, sizeof(triangles));

    const unsigned int flags[1] = { OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT };
    OptixBuildInput input = {};
    input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    input.triangleArray.vertexBuffers = &dev_vertices;
    input.triangleArray.numVertices = 8;
    input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    input.triangleArray.indexBuffer = dev_triangles;
    input.triangleArray.numIndexTriplets = 12;
    input.triangleArray.flags = flags;
    input.triangleArray.numSbtRecords = 1;

    bool ok = buildAccel(input, dev_cubeGas, handle);
    // The build copies what it needs; the inputs are not read at trace time.
    release(dev_vertices);
    release(dev_triangles);
    return ok;
}

// Unit sphere: one custom primitive whose bounds are the unit cube, so
// traversal only knows a box and __intersection__sphere in optix_programs.cu
// does the actual test. The built-in sphere primitive is hollow and never
// hits from inside, which a refractive sphere needs; see the comment there.
bool buildSphereGas(OptixTraversableHandle& handle)
{
    const OptixAabb bounds = { -0.5f, -0.5f, -0.5f, 0.5f, 0.5f, 0.5f };
    CUdeviceptr dev_bounds = upload(&bounds, sizeof(bounds));

    const unsigned int flags[1] = { OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT };
    OptixBuildInput input = {};
    input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    input.customPrimitiveArray.aabbBuffers = &dev_bounds;
    input.customPrimitiveArray.numPrimitives = 1;
    input.customPrimitiveArray.flags = flags;
    input.customPrimitiveArray.numSbtRecords = 1;

    bool ok = buildAccel(input, dev_sphereGas, handle);
    release(dev_bounds);
    return ok;
}

// One triangle GAS per TriangleMesh, from the flat device arrays pathtrace.cu
// uploaded: the whole vertex array serves as the vertex buffer and the mesh's
// slice of the index array as the index buffer. Both faces stay hittable
// (no culling flag), which a refractive mesh needs for the exit hit. A mesh
// keeps any-hit only if some instance of it has an ALPHA_MASK or ALPHA_BLEND material;
// buildIas then turns it off again on the instances that do not.
bool buildMeshGases(const Scene* scene, const MeshBuffers& buffers, std::vector<OptixTraversableHandle>& handles)
{
    dev_meshGas.assign(scene->meshes.size(), 0);
    handles.assign(scene->meshes.size(), 0);
    std::vector<bool> alphaTested(scene->meshes.size(), false);
    for (const Geom& g : scene->geoms)
    {
        if (isAlphaTested(scene, g))
        {
            alphaTested[g.meshId] = true;
        }
    }
    CUdeviceptr dev_vertices = reinterpret_cast<CUdeviceptr>(buffers.positions);
    for (size_t i = 0; i < scene->meshes.size(); ++i)
    {
        const unsigned int flags[1] = {
            static_cast<unsigned int>(alphaTested[i] ? OPTIX_GEOMETRY_FLAG_NONE : OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT) };
        const TriangleMesh& mesh = scene->meshes[i];
        OptixBuildInput input = {};
        input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        input.triangleArray.vertexBuffers = &dev_vertices;
        input.triangleArray.numVertices = static_cast<unsigned int>(scene->positions.size());
        // glm::ivec3 holds non-negative indices, so it reads as uint3.
        input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        input.triangleArray.indexBuffer = reinterpret_cast<CUdeviceptr>(buffers.indices + mesh.indexOffset);
        input.triangleArray.numIndexTriplets = static_cast<unsigned int>(mesh.triCount);
        input.triangleArray.flags = flags;
        input.triangleArray.numSbtRecords = 1;
        if (!buildAccel(input, dev_meshGas[i], handles[i]))
        {
            return false;
        }
    }
    return true;
}

// One instance per Geom: the unit cube, unit sphere or one of the mesh GASes
// under the Geom's transform. instanceId is the Geom's index, which the
// hit programs use to look up the InstanceRecord, and sbtOffset picks the hit
// program (hit group record 0, 1 or 2). Only alpha-tested instances run the
// any-hit program; the flag disables it on every other instance, which may
// share a mesh GAS with an alpha-tested one.
bool buildIas(const Scene* scene, OptixTraversableHandle cubeGas, OptixTraversableHandle sphereGas,
              const std::vector<OptixTraversableHandle>& meshGas)
{
    std::vector<OptixInstance> instances(scene->geoms.size());  // zeroed
    for (size_t i = 0; i < instances.size(); ++i)
    {
        const Geom& g = scene->geoms[i];
        OptixInstance& inst = instances[i];
        // OptiX wants the top three rows of the 4x4, row-major; glm stores columns.
        for (int row = 0; row < 3; ++row)
        {
            for (int col = 0; col < 4; ++col)
            {
                inst.transform[row * 4 + col] = g.transform[col][row];
            }
        }
        inst.instanceId = static_cast<unsigned int>(i);
        inst.visibilityMask = 255;
        inst.flags = isAlphaTested(scene, g) ? OPTIX_INSTANCE_FLAG_NONE : OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT;
        switch (g.type)
        {
        case GeomType::SPHERE:
            inst.sbtOffset = 1;
            inst.traversableHandle = sphereGas;
            break;
        case GeomType::MESH:
            inst.sbtOffset = 2;
            inst.traversableHandle = meshGas[g.meshId];
            break;
        default:
            inst.sbtOffset = 0;
            inst.traversableHandle = cubeGas;
            break;
        }
    }
    CUdeviceptr dev_optixInstances = upload(instances.data(), instances.size() * sizeof(OptixInstance));

    OptixBuildInput input = {};
    input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    input.instanceArray.instances = dev_optixInstances;
    input.instanceArray.numInstances = static_cast<unsigned int>(instances.size());

    bool ok = buildAccel(input, dev_ias, iasHandle);
    release(dev_optixInstances);
    return ok;
}

// Module from the embedded OptiX-IR, six program groups, one pipeline.
bool buildPipeline()
{
    OptixModuleCompileOptions moduleOptions = {};  // defaults: full optimization, line info
    OptixPipelineCompileOptions pipelineOptions = {};
    pipelineOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipelineOptions.numPayloadValues = 0;    // the programs write to params.intersections directly
    pipelineOptions.numAttributeValues = 4;  // __intersection__sphere reports four; triangles need two
    pipelineOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineOptions.pipelineLaunchParamsVariableName = "params";
    pipelineOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE | OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;

    char log[2048];
    size_t logSize = sizeof(log);
    OptixResult result = optixModuleCreate(context, &moduleOptions, &pipelineOptions,
        reinterpret_cast<const char*>(optix_programs_ir), optix_programs_ir_size,
        log, &logSize, &module);
    printLog("module", result, log, logSize);
    if (!check(result, "optixModuleCreate"))
    {
        return false;
    }

    OptixProgramGroupDesc descs[NUM_GROUPS] = {};
    descs[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    descs[0].raygen.module = module;
    descs[0].raygen.entryFunctionName = "__raygen__paths";
    descs[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    descs[1].miss.module = module;
    descs[1].miss.entryFunctionName = "__miss__paths";
    descs[2].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    descs[2].miss.module = module;
    descs[2].miss.entryFunctionName = "__miss__shadow";
    descs[3].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    descs[3].hitgroup.moduleCH = module;
    descs[3].hitgroup.entryFunctionNameCH = "__closesthit__cube";
    descs[4].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    descs[4].hitgroup.moduleCH = module;
    descs[4].hitgroup.entryFunctionNameCH = "__closesthit__sphere";
    descs[4].hitgroup.moduleIS = module;
    descs[4].hitgroup.entryFunctionNameIS = "__intersection__sphere";
    descs[5].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;  // built-in triangle intersection, no IS
    descs[5].hitgroup.moduleCH = module;
    descs[5].hitgroup.entryFunctionNameCH = "__closesthit__mesh";
    descs[5].hitgroup.moduleAH = module;
    descs[5].hitgroup.entryFunctionNameAH = "__anyhit__mesh";
    OptixProgramGroupOptions groupOptions = {};
    logSize = sizeof(log);
    result = optixProgramGroupCreate(context, descs, NUM_GROUPS, &groupOptions, log, &logSize, groups);
    printLog("program group", result, log, logSize);
    if (!check(result, "optixProgramGroupCreate"))
    {
        return false;
    }

    OptixPipelineLinkOptions linkOptions = {};
    linkOptions.maxTraceDepth = 1;  // raygen traces once; nothing traces from a hit
    logSize = sizeof(log);
    result = optixPipelineCreate(context, &pipelineOptions, &linkOptions, groups, NUM_GROUPS,
        log, &logSize, &pipeline);
    printLog("pipeline", result, log, logSize);
    if (!check(result, "optixPipelineCreate"))
    {
        return false;
    }

    // Stack for one trace through an IAS over GASes (traversable graph depth 2).
    OptixStackSizes stackSizes = {};
    for (OptixProgramGroup group : groups)
    {
        OPTIX_TRY(optixUtilAccumulateStackSizes(group, &stackSizes, pipeline));
    }
    unsigned int directCallableFromTraversal = 0;
    unsigned int directCallableFromState = 0;
    unsigned int continuation = 0;
    OPTIX_TRY(optixUtilComputeStackSizes(&stackSizes, 1, 0, 0,
        &directCallableFromTraversal, &directCallableFromState, &continuation));
    OPTIX_TRY(optixPipelineSetStackSize(pipeline, directCallableFromTraversal,
        directCallableFromState, continuation, 2));
    return true;
}

// Records carry only the program header: the programs get everything through
// the launch parameters, and an instance's sbtOffset picks its hit record.
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) Record
{
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

bool buildSbt()
{
    Record records[NUM_GROUPS];
    for (int i = 0; i < NUM_GROUPS; ++i)
    {
        OPTIX_TRY(optixSbtRecordPackHeader(groups[i], &records[i]));
    }
    dev_sbtRecords = upload(records, sizeof(records));

    sbt.raygenRecord = dev_sbtRecords;
    sbt.missRecordBase = dev_sbtRecords + sizeof(Record);
    sbt.missRecordStrideInBytes = sizeof(Record);
    sbt.missRecordCount = NUM_MISS_GROUPS;  // path rays, shadow rays
    sbt.hitgroupRecordBase = dev_sbtRecords + (1 + NUM_MISS_GROUPS) * sizeof(Record);
    sbt.hitgroupRecordStrideInBytes = sizeof(Record);
    sbt.hitgroupRecordCount = NUM_HIT_GROUPS;  // cube, sphere, mesh
    return true;
}

bool init(const Scene* scene, const MeshBuffers& buffers, const Material* materials,
          const cudaTextureObject_t* textures, bool validation)
{
    {
        TimingScope timing("init.optix_context");
        // optixInit loads the driver's OptiX library and fills the function
        // table; the version check against OPTIX_ABI_VERSION happens here.
        OPTIX_TRY(optixInit());

        OptixDeviceContextOptions options = {};
        options.logCallbackFunction = logCallback;
        options.logCallbackLevel = 3;  // fatal, error, warning
        options.validationMode = validation ? OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL
                                            : OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF;
        // A null CUDA context means the current one, which pathtraceInit's
        // allocations have already created, so OptiX shares it with the kernels.
        OPTIX_TRY(optixDeviceContextCreate(nullptr, &options, &context));
    }

    {
        TimingScope timing("init.optix_accel");
        OptixTraversableHandle cubeGas = 0;
        OptixTraversableHandle sphereGas = 0;
        std::vector<OptixTraversableHandle> meshGas;
        if (!buildCubeGas(cubeGas) || !buildSphereGas(sphereGas) || !buildMeshGases(scene, buffers, meshGas)
            || !buildIas(scene, cubeGas, sphereGas, meshGas))
        {
            return false;
        }
        CUDA_CHECK(cudaDeviceSynchronize());  // the builds run on the device; time them, not their launch
    }
    {
        TimingScope timing("init.optix_pipeline");
        if (!buildPipeline() || !buildSbt())
        {
            return false;
        }
    }

    // Per-instance table for the hit programs: one record per Geom, in Geom order.
    std::vector<InstanceRecord> records(scene->geoms.size());
    for (size_t i = 0; i < records.size(); ++i)
    {
        records[i].materialId = scene->geoms[i].materialId;
        records[i].meshId = scene->geoms[i].meshId;
        records[i].tangentSign = scene->geoms[i].tangentSign;
        records[i].lightMask = scene->geoms[i].lightMask;
        records[i].lightGroup = scene->geoms[i].lightGroup;
    }
    dev_instances = reinterpret_cast<InstanceRecord*>(upload(records.data(), records.size() * sizeof(InstanceRecord)));
    meshBuffers = buffers;
    dev_materials = materials;
    dev_textures = textures;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_params), sizeof(OptixIntersectParams)));
    return true;
}
}  // namespace

bool optixIntersectInit(const Scene* scene, const MeshBuffers& buffers, const Material* materials,
                        const cudaTextureObject_t* textures, bool validation)
{
    if (init(scene, buffers, materials, textures, validation))
    {
        return true;
    }
    optixIntersectFree();  // release whatever a partial init created
    return false;
}

void optixIntersectFree()
{
    // Dependents first: pipeline, then program groups, then modules, then the context.
    if (pipeline)
    {
        optixPipelineDestroy(pipeline);
        pipeline = nullptr;
    }
    for (OptixProgramGroup& group : groups)
    {
        if (group)
        {
            optixProgramGroupDestroy(group);
            group = nullptr;
        }
    }
    if (module)
    {
        optixModuleDestroy(module);
        module = nullptr;
    }
    if (context)
    {
        optixDeviceContextDestroy(context);
        context = nullptr;
    }
    release(dev_sbtRecords);
    release(dev_cubeGas);
    release(dev_sphereGas);
    for (CUdeviceptr& gas : dev_meshGas)
    {
        release(gas);
    }
    dev_meshGas.clear();
    release(dev_ias);
    cudaFree(dev_params);
    dev_params = nullptr;
    cudaFree(dev_instances);
    dev_instances = nullptr;
    meshBuffers = {};  // these three are owned by pathtrace.cu, not freed here
    dev_materials = nullptr;
    dev_textures = nullptr;
    iasHandle = 0;
    sbt = {};
}

void optixIntersect(int iter, int numPaths, const PathSegment* paths,
                    ShadeableIntersection* intersections, int* materialIds,
                    int numMaterials, const ShadowRay* shadowRays, const int* shadowCount,
                    int shadowBound, glm::vec3* image, cudaStream_t stream)
{
    if (numPaths + shadowBound <= 0)
    {
        return;
    }

    OptixIntersectParams params = {};
    params.iter = iter;
    params.numPaths = numPaths;
    params.paths = paths;
    params.shadowRays = shadowRays;
    params.shadowCount = shadowCount;
    params.image = image;
    params.intersections = intersections;
    params.materialIds = materialIds;
    params.numMaterials = numMaterials;
    params.instances = dev_instances;
    params.buffers = meshBuffers;
    params.materials = dev_materials;
    params.textures = dev_textures;
    params.handle = iasHandle;
    // The buffers ping-pong every bounce, so the parameters go up per launch.
    // The source is pageable, so the copy is staged before this returns and
    // the stack variable can go out of scope.
    CUDA_CHECK(cudaMemcpyAsync(dev_params, &params, sizeof(params), cudaMemcpyHostToDevice, stream));

    OptixResult result = optixLaunch(pipeline, stream, reinterpret_cast<CUdeviceptr>(dev_params),
        sizeof(OptixIntersectParams), &sbt, numPaths + shadowBound, 1, 1);
    if (result != OPTIX_SUCCESS)
    {
        fatal("optixLaunch failed: %s (%s)", optixGetErrorName(result), optixGetErrorString(result));
    }
}
