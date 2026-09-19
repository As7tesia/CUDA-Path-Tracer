#pragma once

#include "sceneStructs.h"

// Device-wide operations on the wavefront's path array (stream compaction now;
// material sort etc. later). Kept in their own translation unit for build
// time only: a thrust/CUB algorithm instantiation costs ~10 s of nvcc, and
// pathtrace.cu is the file edited most often. Prefer adding new device-wide
// ops here rather than pulling thrust algorithm headers into pathtrace.cu.

// Stable-partitions paths so that every path with remainingBounces > 0 is at
// the front, in original order. Terminated paths (with their final colour)
// follow, so finalGather can still read all of them. Returns the number of
// live paths. Runs on the legacy default stream (thrust::device).
int compactPaths(PathSegment* paths, int numPaths);
