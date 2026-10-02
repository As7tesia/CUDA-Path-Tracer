// OptiX intersection stage, host side. See optix_intersect.h.
//
// A .cpp on purpose: optix.h switches to the device API whenever __CUDACC__
// is defined, so host code that includes it cannot live in a .cu. The CUDA
// runtime calls below only need cuda_runtime.h.

#define NOMINMAX  // optix_stubs.h includes windows.h, whose min/max macros break glm

#include "optix_intersect.h"

#include "optix_params.h"
#include "scene.h"

#include <optix.h>
#include <optix_function_table_definition.h>  // exactly one translation unit defines the table
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

// The device programs from optix_programs.cu, compiled to OptiX-IR by nvcc
// and turned into this array by cmake/EmbedFile.cmake (see CMakeLists.txt).
extern const unsigned char optix_programs_ir[];
extern const size_t optix_programs_ir_size;

namespace
{
// Everything OptiX owns for the lifetime of the scene, created by
// optixIntersectInit and released by optixIntersectFree.
constexpr int kNumGroups = 5;        // raygen, miss, cube hit, sphere hit, mesh hit
constexpr int kNumHitGroups = 3;     // sbtOffset 0 = cube, 1 = sphere, 2 = mesh

OptixDeviceContext context = nullptr;
OptixModule module = nullptr;        // every program: raygen, miss, sphere intersection, closest hits
OptixProgramGroup groups[kNumGroups] = {};
OptixPipeline pipeline = nullptr;
OptixShaderBindingTable sbt = {};
CUdeviceptr d_sbtRecords = 0;
CUdeviceptr d_cubeGas = 0;
CUdeviceptr d_sphereGas = 0;
std::vector<CUdeviceptr> d_meshGas;  // one GAS per TriangleMesh
CUdeviceptr d_ias = 0;
OptixTraversableHandle iasHandle = 0;
OptixIntersectParams* d_params = nullptr;
InstanceRecord* d_instances = nullptr;  // instance id -> material and mesh
// pathtrace.cu's device arrays, passed through to the hit programs
MeshBuffers meshBuffers = {};
const Material* d_materials = nullptr;
const cudaTextureObject_t* d_textures = nullptr;

// Whether a Geom's hits go through the any-hit alpha test.
bool isMasked(const Scene* scene, const Geom& g)
{
    return g.type == MESH && scene->materials[g.materialid].alphaMode == ALPHA_MASK;
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
    cudaMalloc(&d, bytes);
    cudaMemcpy(d, data, bytes, cudaMemcpyHostToDevice);
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
    options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;    // prefer trace speed > build speed, theres another flag that does opposite
    options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes sizes = {};
    OPTIX_TRY(optixAccelComputeMemoryUsage(context, &options, &input, 1, &sizes));

    CUdeviceptr temp = 0;
    cudaMalloc(reinterpret_cast<void**>(&temp), sizes.tempSizeInBytes);
    cudaMalloc(reinterpret_cast<void**>(&output), sizes.outputSizeInBytes);
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
    CUdeviceptr d_vertices = upload(vertices, sizeof(vertices));
    CUdeviceptr d_triangles = upload(triangles, sizeof(triangles));

    const unsigned int flags[1] = { OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT };
    OptixBuildInput input = {};
    input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    input.triangleArray.vertexBuffers = &d_vertices;
    input.triangleArray.numVertices = 8;
    input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    input.triangleArray.indexBuffer = d_triangles;
    input.triangleArray.numIndexTriplets = 12;
    input.triangleArray.flags = flags;
    input.triangleArray.numSbtRecords = 1;

    bool ok = buildAccel(input, d_cubeGas, handle);
    // The build copies what it needs; the inputs are not read at trace time.
    release(d_vertices);
    release(d_triangles);
    return ok;
}

// Unit sphere: one custom primitive whose bounds are the unit cube, so
// traversal only knows a box and __intersection__sphere in optix_programs.cu
// does the actual test. The built-in sphere primitive is hollow and never
// hits from inside, which a refractive sphere needs; see the comment there.
bool buildSphereGas(OptixTraversableHandle& handle)
{
    const OptixAabb bounds = { -0.5f, -0.5f, -0.5f, 0.5f, 0.5f, 0.5f };
    CUdeviceptr d_bounds = upload(&bounds, sizeof(bounds));

    const unsigned int flags[1] = { OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT };
    OptixBuildInput input = {};
    input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    input.customPrimitiveArray.aabbBuffers = &d_bounds;
    input.customPrimitiveArray.numPrimitives = 1;
    input.customPrimitiveArray.flags = flags;
    input.customPrimitiveArray.numSbtRecords = 1;

    bool ok = buildAccel(input, d_sphereGas, handle);
    release(d_bounds);
    return ok;
}

// One triangle GAS per TriangleMesh, from the flat device arrays pathtrace.cu
// uploaded: the whole vertex array serves as the vertex buffer and the mesh's
// slice of the index array as the index buffer. Both faces stay hittable
// (no culling flag), which a refractive mesh needs for the exit hit. A mesh
// keeps any-hit only if some instance of it has an ALPHA_MASK material;
// buildIas then turns it off again on the instances that do not.
bool buildMeshGases(const Scene* scene, const MeshBuffers& buffers, std::vector<OptixTraversableHandle>& handles)
{
    d_meshGas.assign(scene->meshes.size(), 0);
    handles.assign(scene->meshes.size(), 0);
    std::vector<bool> masked(scene->meshes.size(), false);
    for (const Geom& g : scene->geoms)
    {
        if (isMasked(scene, g))
        {
            masked[g.meshId] = true;
        }
    }
    CUdeviceptr d_vertices = reinterpret_cast<CUdeviceptr>(buffers.positions);
    for (size_t i = 0; i < scene->meshes.size(); ++i)
    {
        const unsigned int flags[1] = {
            static_cast<unsigned int>(masked[i] ? OPTIX_GEOMETRY_FLAG_NONE : OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT) };
        const TriangleMesh& mesh = scene->meshes[i];
        OptixBuildInput input = {};
        input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        input.triangleArray.vertexBuffers = &d_vertices;
        input.triangleArray.numVertices = static_cast<unsigned int>(scene->positions.size());
        // glm::ivec3 holds non-negative indices, so it reads as uint3.
        input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        input.triangleArray.indexBuffer = reinterpret_cast<CUdeviceptr>(buffers.indices + mesh.indexOffset);
        input.triangleArray.numIndexTriplets = static_cast<unsigned int>(mesh.triCount);
        input.triangleArray.flags = flags;
        input.triangleArray.numSbtRecords = 1;
        if (!buildAccel(input, d_meshGas[i], handles[i]))
        {
            return false;
        }
    }
    return true;
}

// One instance per Geom: the unit cube, unit sphere or one of the mesh GASes
// under the Geom's transform. instanceId is the Geom's index, which the
// hit programs use to look up the InstanceRecord, and sbtOffset picks the hit
// program (hit group record 0, 1 or 2). Only ALPHA_MASK instances run the
// any-hit program; the flag disables it on every other instance, which may
// share a mesh GAS with a masked one.
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
        inst.flags = isMasked(scene, g) ? OPTIX_INSTANCE_FLAG_NONE : OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT;
        switch (g.type)
        {
        case SPHERE:
            inst.sbtOffset = 1;
            inst.traversableHandle = sphereGas;
            break;
        case MESH:
            inst.sbtOffset = 2;
            inst.traversableHandle = meshGas[g.meshId];
            break;
        default:
            inst.sbtOffset = 0;
            inst.traversableHandle = cubeGas;
            break;
        }
    }
    CUdeviceptr d_instances = upload(instances.data(), instances.size() * sizeof(OptixInstance));

    OptixBuildInput input = {};
    input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    input.instanceArray.instances = d_instances;
    input.instanceArray.numInstances = static_cast<unsigned int>(instances.size());

    bool ok = buildAccel(input, d_ias, iasHandle);
    release(d_instances);
    return ok;
}

// Module from the embedded OptiX-IR, five program groups, one pipeline.
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

    OptixProgramGroupDesc descs[kNumGroups] = {};
    descs[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    descs[0].raygen.module = module;
    descs[0].raygen.entryFunctionName = "__raygen__paths";
    descs[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    descs[1].miss.module = module;
    descs[1].miss.entryFunctionName = "__miss__paths";
    descs[2].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    descs[2].hitgroup.moduleCH = module;
    descs[2].hitgroup.entryFunctionNameCH = "__closesthit__cube";
    descs[3].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    descs[3].hitgroup.moduleCH = module;
    descs[3].hitgroup.entryFunctionNameCH = "__closesthit__sphere";
    descs[3].hitgroup.moduleIS = module;
    descs[3].hitgroup.entryFunctionNameIS = "__intersection__sphere";
    descs[4].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;  // built-in triangle intersection, no IS
    descs[4].hitgroup.moduleCH = module;
    descs[4].hitgroup.entryFunctionNameCH = "__closesthit__mesh";
    descs[4].hitgroup.moduleAH = module;
    descs[4].hitgroup.entryFunctionNameAH = "__anyhit__mesh";
    OptixProgramGroupOptions groupOptions = {};
    logSize = sizeof(log);
    result = optixProgramGroupCreate(context, descs, kNumGroups, &groupOptions, log, &logSize, groups);
    printLog("program group", result, log, logSize);
    if (!check(result, "optixProgramGroupCreate"))
    {
        return false;
    }

    OptixPipelineLinkOptions linkOptions = {};
    linkOptions.maxTraceDepth = 1;  // raygen traces once; nothing traces from a hit
    logSize = sizeof(log);
    result = optixPipelineCreate(context, &pipelineOptions, &linkOptions, groups, kNumGroups,
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
    Record records[kNumGroups];
    for (int i = 0; i < kNumGroups; ++i)
    {
        OPTIX_TRY(optixSbtRecordPackHeader(groups[i], &records[i]));
    }
    d_sbtRecords = upload(records, sizeof(records));

    sbt.raygenRecord = d_sbtRecords;
    sbt.missRecordBase = d_sbtRecords + sizeof(Record);
    sbt.missRecordStrideInBytes = sizeof(Record);
    sbt.missRecordCount = 1;
    sbt.hitgroupRecordBase = d_sbtRecords + 2 * sizeof(Record);
    sbt.hitgroupRecordStrideInBytes = sizeof(Record);
    sbt.hitgroupRecordCount = kNumHitGroups;  // cube, sphere, mesh
    return true;
}

bool init(const Scene* scene, const MeshBuffers& buffers, const Material* materials,
          const cudaTextureObject_t* textures, bool validation)
{
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

    OptixTraversableHandle cubeGas = 0;
    OptixTraversableHandle sphereGas = 0;
    std::vector<OptixTraversableHandle> meshGas;
    if (!buildCubeGas(cubeGas) || !buildSphereGas(sphereGas) || !buildMeshGases(scene, buffers, meshGas)
        || !buildIas(scene, cubeGas, sphereGas, meshGas))
    {
        return false;
    }
    if (!buildPipeline() || !buildSbt())
    {
        return false;
    }

    // Per-instance table for the hit programs: one record per Geom, in Geom order.
    std::vector<InstanceRecord> records(scene->geoms.size());
    for (size_t i = 0; i < records.size(); ++i)
    {
        records[i].materialId = scene->geoms[i].materialid;
        records[i].meshId = scene->geoms[i].meshId;
        records[i].tangentSign = scene->geoms[i].tangentSign;
    }
    d_instances = reinterpret_cast<InstanceRecord*>(upload(records.data(), records.size() * sizeof(InstanceRecord)));
    meshBuffers = buffers;
    d_materials = materials;
    d_textures = textures;
    cudaMalloc(reinterpret_cast<void**>(&d_params), sizeof(OptixIntersectParams));
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
    release(d_sbtRecords);
    release(d_cubeGas);
    release(d_sphereGas);
    for (CUdeviceptr& gas : d_meshGas)
    {
        release(gas);
    }
    d_meshGas.clear();
    release(d_ias);
    cudaFree(d_params);
    d_params = nullptr;
    cudaFree(d_instances);
    d_instances = nullptr;
    meshBuffers = {};  // these three are owned by pathtrace.cu, not freed here
    d_materials = nullptr;
    d_textures = nullptr;
    iasHandle = 0;
    sbt = {};
}

void optixIntersect(int numPaths, const PathSegment* paths,
                    ShadeableIntersection* intersections, int* materialIds,
                    int numMaterials, cudaStream_t stream)
{
    if (numPaths <= 0)
    {
        return;
    }

    OptixIntersectParams params = {};
    params.paths = paths;
    params.intersections = intersections;
    params.materialIds = materialIds;
    params.numMaterials = numMaterials;
    params.instances = d_instances;
    params.buffers = meshBuffers;
    params.materials = d_materials;
    params.textures = d_textures;
    params.handle = iasHandle;
    // The buffers ping-pong every bounce, so the parameters go up per launch.
    // The source is pageable, so the copy is staged before this returns and
    // the stack variable can go out of scope.
    cudaMemcpyAsync(d_params, &params, sizeof(params), cudaMemcpyHostToDevice, stream);

    OptixResult result = optixLaunch(pipeline, stream, reinterpret_cast<CUdeviceptr>(d_params),
        sizeof(OptixIntersectParams), &sbt, numPaths, 1, 1);
    if (!check(result, "optixLaunch"))
    {
        exit(EXIT_FAILURE);
    }
}
