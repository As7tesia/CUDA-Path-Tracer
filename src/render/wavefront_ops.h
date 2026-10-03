#pragma once

#include "scene/sceneStructs.h"

// The wavefront's path buffers and the device-wide operations on them (the
// survivor buffer shadeMaterial compacts into, and the material sort). Kept in
// their own translation unit for build time only: a CUB algorithm
// instantiation costs ~10 s of nvcc, and pathtrace.cu is the file edited most
// often. Prefer adding new device-wide ops here rather than pulling CUB or
// thrust algorithm headers into pathtrace.cu.
//
// Both are out of place and ping-pong: shadeMaterial and the sort write into a
// spare buffer from the workspace below, which then trades places with the
// caller's pointer, so the caller's paths (and, for the sort, intersections)
// point at a different allocation after every call. The spare buffers and
// CUB's temporary storage are shared, so all calls must go to the same stream.

// Threads per block of the 1D kernels over the path array: the naive
// intersection kernel and shadeMaterial in pathtrace.cu, the sort's helper
// kernels here.
constexpr int PATH_BLOCK_SIZE = 128;

// Allocates the workspace for up to maxPaths paths: a spare PathSegment and
// ShadeableIntersection buffer, the sort's index and key arrays, the three
// survivor counters, and CUB temporary storage for the sort. Called from
// pathtraceInit, wavefrontFree from pathtraceFree. The swaps only exchange
// pointers, so pathtrace.cu freeing whatever its pointers hold and
// wavefrontFree freeing whatever the spares hold releases every buffer once.
void wavefrontInit(int maxPaths);
void wavefrontFree();

// Where shadeMaterial appends the paths that survive the current bounce
// (compaction fused into shading): the spare path buffer and the counter of
// this bounce's survivors, which starts at zero. nextCount is the counter the
// following bounce appends into, and the shade kernel zeroes it. Three
// counters rotate by bounce, so the one being zeroed is never one that a
// queued or running kernel still reads. Ask after sortMaterials, which swaps
// the spare too.
struct SurvivorBuffer
{
    PathSegment* paths;
    int* count;
    int* nextCount;
};
SurvivorBuffer wavefrontSurvivors();

// After the shade launch: swaps the survivor buffer with paths, moves the
// counters on to the next bounce and returns the number of survivors. Paths
// that ended are not in the new buffer (shadeMaterial has already added their
// light to the image), and everything past the returned count is stale.
// Reading the count back to the host synchronizes.
int wavefrontSwapPaths(PathSegment*& paths, cudaStream_t stream = 0);

// Sorts the first numPaths paths and their intersections by materialIds (one
// int key per path, written by the intersection stage: materialId on hit,
// numMaterials on miss). Radix-sorts (key, path index) pairs on only the low
// keyBits bits (the caller computes the width of its largest key once, see
// bitsToHold in pathtrace.cu), then gathers paths and intersections into the
// spare buffers in sorted order and swaps both pointers. After this, paths
// hitting the same material are contiguous, so shadeMaterial warps take one
// branch and read one material. materialIds itself is left unsorted.
void sortMaterials(int numPaths, int keyBits, const int* materialIds,
                   ShadeableIntersection*& intersections, PathSegment*& paths,
                   cudaStream_t stream = 0);
