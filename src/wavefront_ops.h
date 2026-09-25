#pragma once

#include "sceneStructs.h"

// Device-wide operations on the wavefront's path array (stream compaction and
// material sort). Kept in their own translation unit for build time only: a
// CUB algorithm instantiation costs ~10 s of nvcc, and pathtrace.cu is the
// file edited most often. Prefer adding new device-wide ops here rather than
// pulling CUB or thrust algorithm headers into pathtrace.cu.
//
// Both ops are out of place and ping-pong: they write into a spare buffer from
// the workspace below and then swap it with the caller's pointer, so the
// caller's paths (and, for the sort, intersections) point at a different
// allocation after every call. The spare buffers and CUB's temporary storage
// are shared by both ops, so all calls must go to the same stream.

// Allocates the workspace for up to maxPaths paths: a spare PathSegment and
// ShadeableIntersection buffer, the sort's index and key arrays, the live
// count, and CUB temporary storage sized once for both algorithms. Called from
// pathtraceInit, wavefrontFree from pathtraceFree. The swaps only exchange
// pointers, so pathtrace.cu freeing whatever its pointers hold and
// wavefrontFree freeing whatever the spares hold releases every buffer once.
void wavefrontInit(int maxPaths);
void wavefrontFree();

// Copies the paths with remainingBounces > 0 from paths[0, numPaths) to the
// front of the spare buffer, in their original order, swaps it with paths and
// returns the number of live paths. Terminated paths are dropped (shadeMaterial
// has already added their color to the image), so everything in paths past the
// returned count is stale. Reading the count back to the host synchronizes.
int compactPaths(PathSegment*& paths, int numPaths, cudaStream_t stream = 0);

// Sorts the first numPaths paths and their intersections by materialIds (one
// int key per path, written by computeIntersections: materialId on hit,
// numMaterials on miss). Radix-sorts (key, path index) pairs on only the low
// keyBits bits (the caller computes the width of its largest key once, see
// bitsToHold in pathtrace.cu), then gathers paths and intersections into the
// spare buffers in sorted order and swaps both pointers. After this, paths
// hitting the same material are contiguous, so shadeMaterial warps take one
// branch and read one material. materialIds itself is left unsorted.
void sortMaterials(int numPaths, int keyBits, const int* materialIds,
                   ShadeableIntersection*& intersections, PathSegment*& paths,
                   cudaStream_t stream = 0);
