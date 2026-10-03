#include "render/wavefront_ops.h"

#include "utilities.h"

#include <cub/device/device_radix_sort.cuh>

#include <utility>

namespace
{
// Workspace, allocated by wavefrontInit. The spare buffers trade places with
// the caller's buffers on every swap (ping pong)
int capacity = 0;
PathSegment* dev_sparePaths = nullptr;
ShadeableIntersection* dev_spareIntersections = nullptr;
int* dev_indices = nullptr;        // 0, 1, 2, ... as the sort's values
int* dev_sortedIndices = nullptr;  // source index of each sorted position
int* dev_sortedKeys = nullptr;     // required by CUB, unused afterwards
void* dev_cubTemp = nullptr;
size_t cubTempBytes = 0;

// Three survivor counters. The current bounce appends into
// dev_survivorCounts[survivorSlot] and zeroes the next slot; the slot moves on
// by one per bounce and carries over from one iteration to the next, so no
// reset is needed between iterations.
int* dev_survivorCounts = nullptr;
int survivorSlot = 0;

// Also catches a missing wavefrontInit: CUB treats a null temp pointer as a
// size query and would return success without doing any work.
void checkCapacity(int numPaths)
{
    if (numPaths > capacity)
    {
        fatal("wavefront_ops: %d paths, the workspace holds %d", numPaths, capacity);
    }
}

__global__ void fillIndices(int n, int* indices)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        indices[i] = i;
    }
}

// Position i of the outputs receives element sortedIndices[i] of the inputs.
__global__ void gatherByIndex(int n, const int* sortedIndices,
    const PathSegment* pathsIn, const ShadeableIntersection* intersectionsIn,
    PathSegment* pathsOut, ShadeableIntersection* intersectionsOut)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        int src = sortedIndices[i];
        pathsOut[i] = pathsIn[src];
        intersectionsOut[i] = intersectionsIn[src];
    }
}
}  // namespace

void wavefrontInit(int maxPaths)
{
    capacity = maxPaths;
    CUDA_CHECK(cudaMalloc(&dev_sparePaths, maxPaths * sizeof(PathSegment)));
    CUDA_CHECK(cudaMalloc(&dev_spareIntersections, maxPaths * sizeof(ShadeableIntersection)));
    CUDA_CHECK(cudaMalloc(&dev_indices, maxPaths * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dev_sortedIndices, maxPaths * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dev_sortedKeys, maxPaths * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dev_survivorCounts, 3 * sizeof(int)));
    CUDA_CHECK(cudaMemset(dev_survivorCounts, 0, 3 * sizeof(int)));
    survivorSlot = 0;

    // With a null temp pointer CUB only reports how much temporary storage it
    // needs, here for the sort at the largest input.
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, cubTempBytes, dev_sortedKeys, dev_sortedKeys,
        dev_indices, dev_sortedIndices, maxPaths));
    CUDA_CHECK(cudaMalloc(&dev_cubTemp, cubTempBytes));
}

void wavefrontFree()
{
    cudaFree(dev_sparePaths);
    cudaFree(dev_spareIntersections);
    cudaFree(dev_indices);
    cudaFree(dev_sortedIndices);
    cudaFree(dev_sortedKeys);
    cudaFree(dev_survivorCounts);
    cudaFree(dev_cubTemp);

    dev_sparePaths = nullptr;
    dev_spareIntersections = nullptr;
    dev_indices = nullptr;
    dev_sortedIndices = nullptr;
    dev_sortedKeys = nullptr;
    dev_survivorCounts = nullptr;
    survivorSlot = 0;
    dev_cubTemp = nullptr;
    cubTempBytes = 0;
    capacity = 0;
}

SurvivorBuffer wavefrontSurvivors()
{
    return { dev_sparePaths, dev_survivorCounts + survivorSlot, dev_survivorCounts + (survivorSlot + 1) % 3 };
}

int wavefrontSwapPaths(PathSegment*& paths, cudaStream_t stream)
{
    // The host loop needs the count to size the next launches, so this is the
    // one synchronization per bounce.
    int numAlive = 0;
    CUDA_CHECK(cudaMemcpyAsync(&numAlive, dev_survivorCounts + survivorSlot, sizeof(int),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::swap(paths, dev_sparePaths);
    survivorSlot = (survivorSlot + 1) % 3;
    return numAlive;
}

void sortMaterials(int numPaths, int keyBits, const int* materialIds,
                   ShadeableIntersection*& intersections, PathSegment*& paths,
                   cudaStream_t stream)
{
    checkCapacity(numPaths);

    const int numBlocks = (numPaths + PATH_BLOCK_SIZE - 1) / PATH_BLOCK_SIZE;
    // 0, 1, 2, ... into dev_indices, the values the sort carries along
    fillIndices<<<numBlocks, PATH_BLOCK_SIZE, 0, stream>>>(numPaths, dev_indices);

    // Only the low keyBits bits take part: one radix pass instead of four for
    // a small scene, because CUB's radix sorts 8 bits per pass and sorting
    // bits 0-31 would take 4 passes.
    size_t bytes = cubTempBytes;
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(dev_cubTemp, bytes,
        materialIds, dev_sortedKeys, dev_indices, dev_sortedIndices,
        numPaths, 0, keyBits, stream));

    // Move the paths themselves, so shadeMaterial's reads stay coalesced.
    gatherByIndex<<<numBlocks, PATH_BLOCK_SIZE, 0, stream>>>(numPaths, dev_sortedIndices,
        paths, intersections, dev_sparePaths, dev_spareIntersections);

    std::swap(paths, dev_sparePaths);
    std::swap(intersections, dev_spareIntersections);
}
