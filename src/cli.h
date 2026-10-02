#pragma once

#include "scene.h"
#include "tonemap.h"

#include <string>

// What the command line asks for. The usage text in cli.cpp lists the flags.
struct Options
{
    std::string sceneFile;
    SceneOverrides overrides;    // --res, --spp, --depth
    bool headless = false;
    std::string outPath;         // empty: img/auto_saved/<FILE>.<time>.<spp>samp.png
    ToneMapMode toneMap = TONEMAP_AGX;
    float exposure = 1.0f;
    bool russianRoulette = true;
    bool materialSort = true;
    bool optix = true;
    bool optixValidation = false;
};

// Reads the arguments. An unknown option, a flag without its value or a value
// that does not parse ends the program with the reason (and the usage text
// for an unknown option or a missing scene).
Options parseArguments(int argc, char** argv);
