#include "wavefront_ops.h"

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_select.cuh>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <utility>

namespace
{
// Predicate for stream compaction: keep paths that still have bounces left.
struct IsAlive
{
    __host__ __device__ bool operator()(const PathSegment& p) const
    {
        return p.remainingBounces > 0;
    }
};

const int blockSize = 128;

// Workspace, allocated by wavefrontInit. The spare buffers trade places with
// the caller's buffers on every swap (ping pong)
int capacity = 0;
PathSegment* dev_sparePaths = nullptr;
ShadeableIntersection* dev_spareIntersections = nullptr;
int* dev_indices = nullptr;        // 0, 1, 2, ... as the sort's values
int* dev_sortedIndices = nullptr;  // source index of each sorted position
int* dev_sortedKeys = nullptr;     // required by CUB, unused afterwards
int* dev_numAlive = nullptr;       // compaction's output count
void* dev_cubTemp = nullptr;
size_t cubTempBytes = 0;

void check(bool ok, const char* msg)
{
    if (!ok)
    {
        fprintf(stderr, "wavefront_ops: %s\n", msg);
        exit(EXIT_FAILURE);
    }
}

// Also catches a missing wavefrontInit: CUB treats a null temp pointer as a
// size query and would return success without doing any work.
void checkCapacity(int numPaths)
{
    check(numPaths <= capacity, "more paths than the workspace was sized for");
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
}
// allocate the scratch space memory cub algos need
void wavefrontInit(int maxPaths)
{
    capacity = maxPaths;
    cudaMalloc(&dev_sparePaths, maxPaths * sizeof(PathSegment));
    cudaMalloc(&dev_spareIntersections, maxPaths * sizeof(ShadeableIntersection));
    cudaMalloc(&dev_indices, maxPaths * sizeof(int));
    cudaMalloc(&dev_sortedIndices, maxPaths * sizeof(int));
    cudaMalloc(&dev_sortedKeys, maxPaths * sizeof(int));
    cudaMalloc(&dev_numAlive, sizeof(int));

    // With a null temp pointer CUB only reports how much temporary storage it
    // needs. Ask both algorithms at the largest input and keep the larger
    // size.
    size_t selectBytes = 0;
    cub::DeviceSelect::If(nullptr, selectBytes, dev_sparePaths, dev_sparePaths,
        dev_numAlive, maxPaths, IsAlive());
    size_t sortBytes = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, sortBytes, dev_sortedKeys, dev_sortedKeys,
        dev_indices, dev_sortedIndices, maxPaths);
    cubTempBytes = std::max(selectBytes, sortBytes);
    cudaMalloc(&dev_cubTemp, cubTempBytes);
}

void wavefrontFree()
{
    cudaFree(dev_sparePaths);
    cudaFree(dev_spareIntersections);
    cudaFree(dev_indices);
    cudaFree(dev_sortedIndices);
    cudaFree(dev_sortedKeys);
    cudaFree(dev_numAlive);
    cudaFree(dev_cubTemp);

    dev_sparePaths = nullptr;
    dev_spareIntersections = nullptr;
    dev_indices = nullptr;
    dev_sortedIndices = nullptr;
    dev_sortedKeys = nullptr;
    dev_numAlive = nullptr;
    dev_cubTemp = nullptr;
    cubTempBytes = 0;
    capacity = 0;
}


int compactPaths(PathSegment*& paths, int numPaths, cudaStream_t stream)
{
    checkCapacity(numPaths);

    size_t bytes = cubTempBytes;
    // copies every path where isAlive is true froim paths to front of dev_sparepaths
    // and writes the count to dev_numAlive
    cudaError_t err = cub::DeviceSelect::If(dev_cubTemp, bytes, paths, dev_sparePaths,
        dev_numAlive, numPaths, IsAlive(), stream);
    check(err == cudaSuccess, cudaGetErrorString(err));

    // The host loop needs the count to size the next launches, so this is the
    // one synchronization per bounce that cannot be avoided.
    int numAlive = 0;
    // copy the count of alive paths to host
    cudaMemcpyAsync(&numAlive, dev_numAlive, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    std::swap(paths, dev_sparePaths);
    return numAlive;
}

void sortMaterials(int numPaths, int keyBits, const int* materialIds,
                   ShadeableIntersection*& intersections, PathSegment*& paths,
                   cudaStream_t stream)
{
    checkCapacity(numPaths);

    const int numBlocks = (numPaths + blockSize - 1) / blockSize;
    // write 0, 1, 2, 3.... into dev_indices, used as value in sorting
    fillIndices<<<numBlocks, blockSize, 0, stream>>>(numPaths, dev_indices);

    // Only the low keyBits bits take part: one radix pass instead of four for
    // a small scene, because CUB's radix sorts 8 bits per pass and sorting
    // bits 0-31 would take 4 passes.
    size_t bytes = cubTempBytes;
    cudaError_t err = cub::DeviceRadixSort::SortPairs(dev_cubTemp, bytes,
        materialIds, dev_sortedKeys, dev_indices, dev_sortedIndices,
        numPaths, 0, keyBits, stream);
    check(err == cudaSuccess, cudaGetErrorString(err));

    // Move the paths themselves, so shadeMaterial's reads stay coalesced.
    gatherByIndex<<<numBlocks, blockSize, 0, stream>>>(numPaths, dev_sortedIndices,
        paths, intersections, dev_sparePaths, dev_spareIntersections);

    std::swap(paths, dev_sparePaths);
    std::swap(intersections, dev_spareIntersections);
}
