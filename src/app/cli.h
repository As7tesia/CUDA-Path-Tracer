#pragma once

#include "scene/scene.h"
#include "render/tonemap.h"

#include <string>

// What the command line asks for. The usage text in cli.cpp lists the flags.
struct Options
{
    std::string sceneFile;       // a path, or a name for findSceneFile (scene.h)
    bool list = false;           // --list: print the scene names and exit
    SceneOverrides overrides;    // --res, --spp, --depth, --env
    bool headless = false;
    std::string outPath;         // empty: img/auto_saved/<FILE>.<time>.<spp>samp.png
    ToneMapMode toneMap = TONEMAP_AGX_PUNCHY;
    float exposure = 1.0f;
    bool russianRoulette = true;
    // Off by default since the 2026-10-02 profile: the sort's gather costs
    // 2 to 3.5x the frame on every scene and saves the shade kernel almost
    // nothing (README, Performance). --sort turns it on.
    bool materialSort = false;
    bool optix = true;
    bool optixValidation = false;
    bool timing = false;         // --timing: headless prints load, init and per-bounce stage times
};

// Reads the arguments. An unknown option, a flag without its value or a value
// that does not parse ends the program with the reason (and the usage text
// for an unknown option or a missing scene).
Options parseArguments(int argc, char** argv);
