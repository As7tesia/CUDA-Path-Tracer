#include "wavefront_ops.h"

#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/partition.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>

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
}

int compactPaths(PathSegment* paths, int numPaths)
{
    PathSegment* aliveEnd = thrust::stable_partition(
        thrust::device, paths, paths + numPaths, IsAlive());
    return static_cast<int>(aliveEnd - paths);
}

void sortMaterials(int numPaths, int* materialIds,
                   ShadeableIntersection* intersections, PathSegment* paths)
{
    // Sort keys ascending; paths and intersections receive the same permutation
    // so every path stays paired with its own intersection.
    thrust::sort_by_key(
        thrust::device,
        materialIds, materialIds + numPaths,
        thrust::make_zip_iterator(thrust::make_tuple(paths, intersections)));
}
