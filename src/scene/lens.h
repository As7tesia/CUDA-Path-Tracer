#pragma once

// A lens prescription: the table of surfaces a patent or PBRT lens file
// lists, loaded for the real-lens camera. The table is runtime data, read
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
    float ior;             // n_d of what fills the gap behind the surface; 1 = air
    float abbe;            // V_d of that glass; 0 = not known, the glass then has one index for every wavelength
    float apertureRadius;  // half the clear diameter; a ray farther from the axis is blocked
    float conic;           // k in the (1 + k) sag form: 0 is a sphere, -1 a paraboloid
    float aspheric[6];     // A4, A6 ... A14 of the sag polynomial, mm^-3, mm^-5 ...; all zero on a spherical surface
    int isStop;            // 1 on the aperture stop, a flat opening with no glass of its own
};

// A whole lens: its surfaces and what the file says about the design.
struct LensSystem
{
    std::string name;
    std::string source;          // where the numbers came from
    std::string apertureSource;  // where the clear apertures came from; patents rarely print them
    std::vector<LensSurface> surfaces;  // scene side first
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
};

// The lens file an argument names: the argument itself when it is an existing
// file, else scenes/lenses/<argument>, with .json and then .dat tried when
// the name has no extension. Throws a std::runtime_error naming what was
// tried when nothing matches.
std::string findLensFile(const std::string& argument);

// Reads a lens file, found with findLensFile. A .json file is this project's
// own format (scenes/lenses/README.md); a .dat file is PBRT-v3's lens
// format: one surface per line as radius, thickness, index and clear
// diameter, # starting a comment. Throws a std::runtime_error, starting with
// the file's name, when the file cannot be read or is not a lens.
LensSystem loadLens(const std::string& argument);

// The two parsers behind loadLens, on the file's text, for the host test.
// name is used in error messages.
LensSystem parseLensJson(const std::string& text, const std::string& name);
LensSystem parseLensDat(const std::string& text, const std::string& name);

// Front vertex to film along the axis: the sum of the thicknesses. Without
// a known last gap this is the length of the glass alone.
float lensLength(const LensSystem& lens);
