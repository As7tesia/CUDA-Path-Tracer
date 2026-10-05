// Host-side check of the lens loaders (scene/lens.cpp): PBRT's
// dgauss.50mm.dat and this project's .json format read into the same table,
// the Noct file's aspheric and focus fields land where they should, and bad
// files are refused with a reason. Every check prints a line and the exit
// code is nonzero when one fails. Built by the lens_test target
// (CMakeLists.txt, outside the default build); run it from the repository
// root, where scenes/lenses/ is.
#include "scene/lens.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

static int checks = 0;
static int failures = 0;

static void check(bool ok, const char* what)
{
    ++checks;
    failures += ok ? 0 : 1;
    printf("%s  %s\n", ok ? "ok  " : "FAIL", what);
}

static bool near(float a, float b, float tolerance = 1e-4f)
{
    return std::fabs(a - b) <= tolerance;
}

// The double Gauss, from either format: the table the lens notes describe.
static void checkDgauss(const LensSystem& lens, const char* label)
{
    char line[256];
    snprintf(line, sizeof line, "%s: 11 surfaces (%d)", label, (int)lens.surfaces.size());
    check(lens.surfaces.size() == 11, line);
    if (lens.surfaces.size() != 11)
    {
        return;
    }
    snprintf(line, sizeof line, "%s: the stop is surface 6 (%d)", label, lens.stopIndex + 1);
    check(lens.stopIndex == 5 && lens.surfaces[5].isStop && lens.surfaces[5].radius == 0.0f, line);
    int stops = 0;
    for (const LensSurface& s : lens.surfaces)
    {
        stops += s.isStop;
    }
    check(stops == 1, "dgauss: exactly one surface is the stop");
    check(near(lens.surfaces[0].radius, 29.475f) && near(lens.surfaces[0].thickness, 3.76f)
        && near(lens.surfaces[0].ior, 1.67f), "dgauss: first surface r 29.475, d 3.76, n 1.67");
    check(near(lens.surfaces[0].apertureRadius, 12.6f) && near(lens.surfaces[5].apertureRadius, 8.55f),
        "dgauss: apertures are radii, half the file's diameters (12.6 front, 8.55 stop)");
    check(lens.surfaces[1].ior == 1.0f && lens.surfaces[5].ior == 1.0f && lens.surfaces[10].ior == 1.0f,
        "dgauss: air gaps and the stop read as index 1");
    check(near(lens.surfaces[10].radius, -39.73f), "dgauss: last surface r -39.73");
    float glass = 0.0f;
    for (size_t i = 0; i + 1 < lens.surfaces.size(); ++i)
    {
        glass += lens.surfaces[i].thickness;
    }
    snprintf(line, sizeof line, "%s: front vertex to the last surface 32.04 mm (%.3f)", label, glass);
    check(near(glass, 32.04f, 1e-3f), line);
    for (const LensSurface& s : lens.surfaces)
    {
        if (s.conic != 0.0f || s.aspheric[0] != 0.0f || s.aspheric[5] != 0.0f)
        {
            check(false, "dgauss: a surface has aspheric terms");
            return;
        }
    }
    check(true, "dgauss: every surface is spherical");
}

int main()
{
    try
    {
        // The PBRT file as is: no Abbe numbers, film gap 0
        const LensSystem dat = loadLens("dgauss.50mm.dat");
        checkDgauss(dat, "dgauss .dat");
        check(dat.surfaces[10].thickness == 0.0f && near(lensLength(dat), 32.04f, 1e-3f),
            "dgauss .dat: the film gap is 0 (solved by focusing)");
        bool noAbbe = true;
        for (const LensSurface& s : dat.surfaces)
        {
            noAbbe = noAbbe && s.abbe == 0.0f;
        }
        check(noAbbe, "dgauss .dat: no Abbe numbers");
        check(dat.stopIndex == 5, "dgauss .dat: findLensFile resolved the bare name in scenes/lenses/");

        // The .json: the same glass, plus the patent's Abbe numbers and the film distance
        const LensSystem js = loadLens("scenes/lenses/dgauss.50mm.json");
        checkDgauss(js, "dgauss .json");
        const float abbe[11] = {47.1f, 0, 47.1f, 30.1f, 0, 0, 38.4f, 57.0f, 0, 48.1f, 0};
        bool abbeOk = true;
        for (int i = 0; i < 11; ++i)
        {
            abbeOk = abbeOk && near(js.surfaces[i].abbe, abbe[i]);
            // Every glass row has one, every air row none
            abbeOk = abbeOk && ((js.surfaces[i].ior > 1.0f) == (abbe[i] > 0.0f));
        }
        check(abbeOk, "dgauss .json: Abbe numbers on the six glass rows only");
        char line[256];
        snprintf(line, sizeof line, "dgauss .json: film 68.146 mm behind the front vertex (%.3f)", lensLength(js));
        check(near(lensLength(js), 68.146f, 1e-3f), line);
        check(near(js.focalLength, 50.0f) && near(js.fNumber, 2.0f) && js.blades == 0, "dgauss .json: stated f 50, f/2, round stop");
        for (int i = 0; i < 11; ++i)
        {
            if (!near(js.surfaces[i].radius, dat.surfaces[i].radius) || !near(js.surfaces[i].ior, dat.surfaces[i].ior)
                || !near(js.surfaces[i].apertureRadius, dat.surfaces[i].apertureRadius)
                || (i < 10 && !near(js.surfaces[i].thickness, dat.surfaces[i].thickness)))
            {
                snprintf(line, sizeof line, "dgauss: .json surface %d matches the .dat", i + 1);
                check(false, line);
            }
        }
        check(true, "dgauss: the .json's geometry matches the .dat");

        // The Noct: aspheres, a focusing gap, estimated apertures
        const LensSystem noct = loadLens("noct-z58");
        snprintf(line, sizeof line, "noct: 28 surfaces (%d), stop at 14 (%d)", (int)noct.surfaces.size(), noct.stopIndex + 1);
        check(noct.surfaces.size() == 28 && noct.stopIndex == 13, line);
        if (noct.surfaces.size() == 28)
        {
            int aspheric = 0;
            for (int i = 0; i < 28; ++i)
            {
                const bool has = noct.surfaces[i].aspheric[0] != 0.0f;
                aspheric += has;
                const bool expected = i == 0 || i == 19 || i == 27;
                if (has != expected)
                {
                    snprintf(line, sizeof line, "noct: surface %d aspheric", i + 1);
                    check(false, line);
                }
            }
            check(aspheric == 3, "noct: aspheres on surfaces 1, 20 and 28 only");
            check(noct.surfaces[0].conic == 0.0f && near(noct.surfaces[0].aspheric[0], -3.82177e-7f, 1e-12f)
                && near(noct.surfaces[0].aspheric[3], -1.32266e-18f, 1e-23f) && noct.surfaces[0].aspheric[4] == 0.0f,
                "noct: surface 1 k 0, A4 -3.82177e-7, A10 -1.32266e-18, no A12");
            check(near(noct.surfaces[27].aspheric[5], -1.70470e-19f, 1e-24f), "noct: surface 28 has A14");
            check(noct.focusSurface == 21 && near(noct.focusGap.x, 2.68f) && near(noct.focusGap.y, 21.29f)
                && near(noct.surfaces[21].thickness, 2.68f), "noct: surface 22 focuses, 2.68 mm at infinity, 21.29 close, table at infinity");
            check(near(noct.closeFocusMagnification, -0.194f), "noct: close focus magnification -0.194");
            check(near(noct.surfaces[0].ior, 1.90265f) && near(noct.surfaces[0].abbe, 35.77f), "noct: surface 1 n 1.90265, V 35.77");
            check(near(noct.surfaces[13].apertureRadius, 0.5f * 47.918f), "noct: stop radius 23.959");
            snprintf(line, sizeof line, "noct: 160.2 mm to the film (%.2f)", lensLength(noct));
            check(near(lensLength(noct), 160.2f, 1e-2f), line);
            check(noct.blades == 11 && near(noct.sensorMm.x, 36.0f) && near(noct.sensorMm.y, 24.0f)
                && near(noct.focalLength, 59.62f) && near(noct.fNumber, 0.98f), "noct: 11 blades, 36x24 sensor, f 59.62, f/0.98");
            check(noct.apertureSource == "estimated", "noct: apertures marked as estimated");
        }
        check(dat.focusSurface == -1 && js.focusSurface == -1, "dgauss: no focusing gap in either file");
    }
    catch (const std::exception& e)
    {
        printf("FAIL  %s\n", e.what());
        ++checks;
        ++failures;
    }

    // Bad files are refused with the reason
    struct Bad
    {
        const char* what;
        const char* text;
        bool json;
    };
    const Bad bad[] = {
        {"json without a stop", "{\"surfaces\": [{\"radius\": 10, \"thickness\": 1, \"aperture\": 5}, {\"radius\": -10, \"thickness\": 0, \"aperture\": 5}]}", true},
        {"json with two stops", "{\"surfaces\": [{\"radius\": 0, \"thickness\": 1, \"aperture\": 5, \"stop\": true}, {\"radius\": 0, \"thickness\": 0, \"aperture\": 5, \"stop\": true}]}", true},
        {"json with a zero gap inside", "{\"surfaces\": [{\"radius\": 10, \"thickness\": 0, \"aperture\": 5}, {\"radius\": 0, \"thickness\": 0, \"aperture\": 5, \"stop\": true}]}", true},
        {"json with seven aspheric terms", "{\"surfaces\": [{\"radius\": 0, \"thickness\": 1, \"aperture\": 5, \"stop\": true, \"aspheric\": [1,2,3,4,5,6,7]}]}", true},
        {"json missing a radius", "{\"surfaces\": [{\"thickness\": 1, \"aperture\": 5, \"stop\": true}]}", true},
        {"json that is not json", "{surfaces: [", true},
        {"dat with a short row", "# comment\n10 1 1.5\n0 1 0 5\n", false},
        {"dat without a stop", "10 1 1.5 5\n-10 0 1 5\n", false},
    };
    for (const Bad& b : bad)
    {
        bool refused = false;
        std::string why;
        try
        {
            if (b.json)
            {
                parseLensJson(b.text, "test");
            }
            else
            {
                parseLensDat(b.text, "test");
            }
        }
        catch (const std::runtime_error& e)
        {
            refused = true;
            why = e.what();
        }
        char line[512];
        snprintf(line, sizeof line, "refused %s: %s", b.what, why.c_str());
        check(refused, line);
    }
    bool missing = false;
    try
    {
        loadLens("no-such-lens");
    }
    catch (const std::runtime_error&)
    {
        missing = true;
    }
    check(missing, "refused a lens file that does not exist");

    printf("%d of %d checks passed\n", checks - failures, checks);
    return failures == 0 ? 0 : 1;
}
