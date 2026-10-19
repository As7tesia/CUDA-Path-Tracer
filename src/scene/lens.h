#pragma once

// A lens prescription: the table of surfaces a patent lists, loaded for the
// real-lens camera. The table is runtime data, read
// from scenes/lenses/ and uploaded by pathtraceInit; nothing in it is a
// compile-time constant.

#include <glm/glm.hpp>

#include <string>
#include <vector>

// One surface, in millimeters, the way prescriptions list them: the surface
// itself, then the gap behind it and what fills that gap. Surfaces run from
// the scene side to the film. The struct is plain data so a kernel can read
// an array of them.
struct LensSurface
{
    float radius;          // curvature radius; positive bulges toward the scene; 0 = flat (the stop, a plano face)
    float thickness;       // along the axis to the next surface; on the last surface, to the film at infinity focus (0 = not known yet)
    float ior;             // n_d of what fills the gap behind the 1@surface; 1 = air
    float abbe;            // V_d of that glass; 0 = not known, the glass then has one index for every wavelength
    float apertureRadius;  // half the clear diameter; a ray farther from the axis is blocked
    float conic;           // k in the (1 + k) sag form: 0 is a sphere, -1 a paraboloid
    // The sag polynomial's coefficients A4, A6 ... as many as the file
    // lists, in LensSystem::aspheric from asphericOffset on. Count 0 on a
    // spherical surface.
    int asphericOffset;
    int asphericCount;
    int isStop;            // 1 on the aperture stop, a flat opening with no glass of its own
};

// A whole lens: its surfaces and what the file says about the design.
struct LensSystem
{
    std::string name;
    std::string file;            // the path loadLens read it from
    std::string source;          // where the numbers came from
    std::string apertureSource;  // where the clear apertures came from; patents rarely print them
    std::vector<LensSurface> surfaces;  // scene side first
    // Every aspheric surface's A4, A6 ... in a row (mm^-3, mm^-5 ...), the
    // surfaces pointing into it. Empty for an all-spherical lens.
    std::vector<float> aspheric;
    int stopIndex = -1;          // the one surface with isStop
    float focalLength = 0.0f;    // as the file states it, mm; 0 = not stated
    float fNumber = 0.0f;        // wide open, as stated; 0 = not stated
    glm::vec2 sensorMm = glm::vec2(36.0f, 24.0f);  // film width and height
    int blades = 0;              // aperture blades; 0 = a round stop
    // Internal focusing, when the file gives it: the surface whose gap
    // changes, and that gap at infinity and at the closest focus. -1 when
    // the file has none (the film then moves to focus).
    int focusSurface = -1;
    glm::vec2 focusGap = glm::vec2(0.0f);
    float closeFocusMagnification = 0.0f;  // at the closest focus; 0 = not stated

    // What a trace of the table says about the lens (analyzeLens, run by
    // loadLens), all in mm. Checked against the stated numbers in the host
    // test; the renderer reads filmDistance and the entrance pupil.
    struct Paraxial
    {
        float focalLength;          // effective focal length, from a ray close to the axis
        float backFocus;            // last surface to the paraxial focus of a distant object
        float filmDistance;         // front vertex to the film: the table's gaps summed
        float entrancePupilZ;       // the stop's center as seen from the scene, behind the front vertex
        float entrancePupilRadius;  // the widest ray parallel to the axis that gets through
        float fNumber;              // focalLength / (2 entrancePupilRadius), wide open
    } paraxial = {};
};

// Fills lens.paraxial from a trace of the surfaces (render/lens_trace.h),
// and when the last gap is 0 (the film distance not known) sets it so a
// distant object is in focus. Throws a std::runtime_error when a ray along
// the axis cannot get through the lens.
void analyzeLens(LensSystem& lens);

// Where the lens files live, "scenes/lenses/"
extern const std::string LENS_DIR;

// The lens file an argument names: the argument itself when it is an existing
// file, else LENS_DIR/<argument>, with .json added when the name has no
// extension. Throws a std::runtime_error naming what was tried when nothing
// matches.
std::string findLensFile(const std::string& argument);

// The .json files in LENS_DIR, sorted, for the window's lens list. Empty
// when there is no such folder.
std::vector<std::string> lensFiles();

// Reads a lens file, found with findLensFile, and analyzes it. The file is
// this project's .json format (scenes/lenses/README.md). Throws a
// std::runtime_error, starting with the file's name, when the file cannot be
// read or is not a lens.
LensSystem loadLens(const std::string& argument);

// The parser behind loadLens, on the file's text, for the host test. name is
// used in error messages.
LensSystem parseLensJson(const std::string& text, const std::string& name);

// Front vertex to film along the axis: the sum of the thicknesses. Without
// a known last gap this is the length of the glass alone.
float lensLength(const LensSystem& lens);
