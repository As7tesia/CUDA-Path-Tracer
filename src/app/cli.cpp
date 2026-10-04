#include "app/cli.h"

#include "utilities.h"

#include <climits>
#include <cstdio>
#include <cstdlib>

namespace
{
const char* const USAGE =
    "Usage: %s SCENE [--headless] [--spp N] [--res WxH] [--depth N] [--out PATH.png] [--env FILE]\n"
    "                     [--no-rr] [--sort] [--no-nee] [--tonemap none|aces|agx|agx-punchy] [--exposure X]\n"
    "                     [--no-optix] [--optix-validate] [--timing]\n"
    "       %s --list\n"
    "  SCENE            a scene .json, a .gltf / .glb file that is the whole scene, or a scene name:\n"
    "                   scenes/<name>.json or an entry of scenes/catalog.json\n"
    "  --list           print the scene names and exit\n"
    "  --headless       render without a window and exit after saving\n"
    "  --spp N          override the scene's ITERATIONS\n"
    "  --res WxH        override the scene's RES\n"
    "  --depth N        override the scene's DEPTH, the most rays a path may trace\n"
    "  --out PATH       write exactly this file (default: img/auto_saved/<FILE>.<time>.<spp>samp.png);\n"
    "                   a .hdr path keeps the scene-linear average, without view transform or exposure\n"
    "  --env FILE       light the scene with this lat-long .hdr or .exr instead of its own environment\n"
    "  --no-rr          disable Russian roulette path termination\n"
    "  --sort           sort paths by material before shading (off by default: it costs 2 to 3.5x, see the README)\n"
    "  --no-sort        the default, kept for scripts\n"
    "  --no-nee         no next event estimation: lights count only when a path hits them\n"
    "  --no-optix       intersect with the naive per-object kernel instead of OptiX (implies --no-nee)\n"
    "  --optix-validate OptiX validation mode: checks every launch, slow\n"
    "  --timing         with --headless: print load, init and per-bounce stage times as CSV lines\n"
    "  --tonemap MODE   view transform for display and PNG (default agx-punchy; none = raw clamp)\n"
    "  --exposure X     linear multiplier before the view transform (default 1.0)\n";
}  // namespace

Options parseArguments(int argc, char** argv)
{
    Options options;
    for (int i = 1; i < argc; i++)
    {
        std::string a = argv[i];
        auto needValue = [&](const char* flag) -> const char* {
            if (i + 1 >= argc)
            {
                fatal("%s needs a value", flag);
            }
            return argv[++i];
        };
        auto needPositiveInt = [&](const char* flag) -> int {
            const char* value = needValue(flag);
            char* end = nullptr;
            const long n = strtol(value, &end, 10);
            if (end == value || *end != '\0' || n <= 0 || n > INT_MAX)
            {
                fatal("%s expects a positive whole number, not \"%s\"", flag, value);
            }
            return (int)n;
        };
        if (a == "--headless")
        {
            options.headless = true;
        }
        else if (a == "--list")
        {
            options.list = true;
        }
        else if (a == "--spp")
        {
            options.overrides.iterations = needPositiveInt("--spp");
        }
        else if (a == "--res")
        {
            const char* value = needValue("--res");
            SceneOverrides& ov = options.overrides;
            if (sscanf(value, "%dx%d", &ov.width, &ov.height) != 2 || ov.width <= 0 || ov.height <= 0)
            {
                fatal("--res expects WxH, e.g. 800x600, not \"%s\"", value);
            }
        }
        else if (a == "--depth")
        {
            options.overrides.traceDepth = needPositiveInt("--depth");
        }
        else if (a == "--out")
        {
            options.outPath = needValue("--out");
        }
        else if (a == "--env")
        {
            options.overrides.environmentFile = needValue("--env");
        }
        else if (a == "--no-rr")
        {
            options.russianRoulette = false;
        }
        else if (a == "--sort")
        {
            options.materialSort = true;
        }
        else if (a == "--no-sort")
        {
            options.materialSort = false;
        }
        else if (a == "--no-nee")
        {
            options.nee = false;
        }
        else if (a == "--no-optix")
        {
            options.optix = false;
        }
        else if (a == "--optix-validate")
        {
            options.optixValidation = true;
        }
        else if (a == "--timing")
        {
            options.timing = true;
        }
        else if (a == "--tonemap")
        {
            std::string m = needValue("--tonemap");
            if (m == "none")      options.toneMap = TONEMAP_NONE;
            else if (m == "aces") options.toneMap = TONEMAP_ACES;
            else if (m == "agx")  options.toneMap = TONEMAP_AGX;
            else if (m == "agx-punchy") options.toneMap = TONEMAP_AGX_PUNCHY;
            else
            {
                fatal("--tonemap expects none, aces, agx or agx-punchy, not \"%s\"", m.c_str());
            }
        }
        else if (a == "--exposure")
        {
            const char* value = needValue("--exposure");
            char* end = nullptr;
            options.exposure = strtof(value, &end);
            if (end == value || *end != '\0' || !(options.exposure > 0.0f))
            {
                fatal("--exposure expects a positive number, not \"%s\"", value);
            }
        }
        else if (a.size() > 2 && a.substr(0, 2) == "--")
        {
            fprintf(stderr, USAGE, argv[0], argv[0]);
            fatal("unknown option %s", argv[i]);
        }
        else
        {
            options.sceneFile = argv[i];
        }
    }

    if (options.sceneFile.empty() && !options.list)
    {
        fprintf(stderr, USAGE, argv[0], argv[0]);
        fatal("no scene given");
    }
    return options;
}
