#include "wavefront_ops.h"

#include <thrust/execution_policy.h>
#include <thrust/partition.h>

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
