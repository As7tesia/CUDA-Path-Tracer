#include "scene/lens.h"

#include "json.hpp"

#include <cctype>
#include <cstdarg>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

using json = nlohmann::json;

namespace
{
const std::string LENS_DIR = "scenes/lenses/";

[[noreturn]] void lensError(const std::string& name, const char* format, ...)
{
    char message[1024];
    va_list args;
    va_start(args, format);
    vsnprintf(message, sizeof message, format, args);
    va_end(args);
    throw std::runtime_error(name + ": " + message);
}

std::string lowercaseExtensionOf(const std::string& path)
{
    std::string ext = std::filesystem::path(path).extension().string();
    for (char& c : ext)
    {
        c = (char)std::tolower((unsigned char)c);
    }
    return ext;
}

// The checks both formats need once the surfaces are read: one stop, gaps
// that go forward, apertures that let something through.
void validate(LensSystem& lens, const std::string& name)
{
    if (lens.surfaces.empty())
    {
        lensError(name, "no surfaces");
    }
    lens.stopIndex = -1;
    for (size_t i = 0; i < lens.surfaces.size(); ++i)
    {
        const LensSurface& s = lens.surfaces[i];
        if (s.isStop)
        {
            if (lens.stopIndex >= 0)
            {
                lensError(name, "surfaces %d and %d are both the stop", lens.stopIndex + 1, (int)i + 1);
            }
            lens.stopIndex = (int)i;
        }
        // Only the last gap may be 0, meaning the film distance is not known
        const bool last = i + 1 == lens.surfaces.size();
        if (s.thickness < 0.0f || (!last && s.thickness == 0.0f))
        {
            lensError(name, "surface %d has thickness %g", (int)i + 1, s.thickness);
        }
        if (!(s.apertureRadius > 0.0f))
        {
            lensError(name, "surface %d has no clear aperture", (int)i + 1);
        }
        if (s.ior < 1.0f)
        {
            lensError(name, "surface %d has index %g, below air", (int)i + 1, s.ior);
        }
    }
    if (lens.stopIndex < 0)
    {
        lensError(name, "no aperture stop (a surface with radius 0 and index 0 in a .dat, \"stop\": true in a .json)");
    }
}
}  // namespace

std::string findLensFile(const std::string& argument)
{
    if (std::filesystem::is_regular_file(argument))
    {
        return argument;
    }
    const std::string inFolder = LENS_DIR + argument;
    for (const char* ext : {"", ".json", ".dat"})
    {
        if (std::filesystem::is_regular_file(inFolder + ext))
        {
            return inFolder + ext;
        }
    }
    throw std::runtime_error("no lens file " + argument + " (tried " + argument + ", " + inFolder + ", " + inFolder
        + ".json and " + inFolder + ".dat)");
}

LensSystem loadLens(const std::string& argument)
{
    const std::string file = findLensFile(argument);
    std::ifstream in(file, std::ios::binary);
    if (!in)
    {
        throw std::runtime_error(file + ": cannot open");
    }
    std::stringstream text;
    text << in.rdbuf();
    const std::string ext = lowercaseExtensionOf(file);
    if (ext == ".json")
    {
        return parseLensJson(text.str(), file);
    }
    if (ext == ".dat")
    {
        return parseLensDat(text.str(), file);
    }
    throw std::runtime_error(file + ": not a lens file (.json or .dat)");
}

LensSystem parseLensDat(const std::string& text, const std::string& name)
{
    LensSystem lens;
    lens.name = std::filesystem::path(name).stem().string();
    lens.source = "PBRT lens file " + name;
    lens.apertureSource = "the lens file";
    std::istringstream in(text);
    std::string line;
    int lineNumber = 0;
    while (std::getline(in, line))
    {
        ++lineNumber;
        const size_t hash = line.find('#');
        if (hash != std::string::npos)
        {
            line.erase(hash);
        }
        std::istringstream fields(line);
        float radius;
        float thickness;
        float ior;
        float diameter;
        if (!(fields >> radius))
        {
            continue;  // blank or comment only
        }
        if (!(fields >> thickness >> ior >> diameter))
        {
            lensError(name, "line %d: expected radius, thickness, index and aperture diameter", lineNumber);
        }
        LensSurface s = {};
        s.radius = radius;
        s.thickness = thickness;
        // PBRT writes the stop as radius 0 with index 0; air is 1 (or 0 in
        // some files). Either way the gap behind it is air.
        s.isStop = radius == 0.0f && ior == 0.0f;
        s.ior = ior < 1.0f ? 1.0f : ior;
        s.abbe = 0.0f;
        s.apertureRadius = 0.5f * diameter;
        lens.surfaces.push_back(s);
    }
    validate(lens, name);
    return lens;
}

LensSystem parseLensJson(const std::string& text, const std::string& name)
{
    json data;
    try
    {
        data = json::parse(text);
    }
    catch (const std::exception& e)
    {
        lensError(name, "%s", e.what());
    }
    LensSystem lens;
    try
    {
        lens.name = data.value("name", std::filesystem::path(name).stem().string());
        lens.source = data.value("source", name);
        lens.apertureSource = data.value("apertures", std::string("the lens file"));
        lens.focalLength = data.value("focalLength", 0.0f);
        lens.fNumber = data.value("fNumber", 0.0f);
        lens.blades = data.value("blades", 0);
        lens.closeFocusMagnification = data.value("closeFocusMagnification", 0.0f);
        if (data.contains("sensor"))
        {
            const json& sensor = data.at("sensor");
            if (!sensor.is_array() || sensor.size() != 2)
            {
                lensError(name, "sensor is not [width, height]");
            }
            lens.sensorMm = glm::vec2(sensor.at(0).get<float>(), sensor.at(1).get<float>());
        }
        const json& surfaces = data.at("surfaces");
        if (!surfaces.is_array())
        {
            lensError(name, "surfaces is not an array");
        }
        for (size_t i = 0; i < surfaces.size(); ++i)
        {
            const json& j = surfaces.at(i);
            LensSurface s = {};
            s.radius = j.at("radius").get<float>();
            s.thickness = j.at("thickness").get<float>();
            s.ior = j.value("ior", 1.0f);
            s.abbe = j.value("abbe", 0.0f);
            s.apertureRadius = 0.5f * j.at("aperture").get<float>();
            s.isStop = j.value("stop", false) ? 1 : 0;
            s.conic = j.value("conic", 0.0f);
            if (j.contains("aspheric"))
            {
                const json& a = j.at("aspheric");
                if (!a.is_array())
                {
                    lensError(name, "surface %d: aspheric is not an array [A4, A6, ...]", (int)i + 1);
                }
                s.asphericOffset = (int)lens.aspheric.size();
                s.asphericCount = (int)a.size();
                for (size_t k = 0; k < a.size(); ++k)
                {
                    lens.aspheric.push_back(a.at(k).get<float>());
                }
            }
            if (j.contains("focus"))
            {
                const json& f = j.at("focus");
                if (!f.is_array() || f.size() != 2)
                {
                    lensError(name, "surface %d: focus is not [gap at infinity, gap at the closest focus]", (int)i + 1);
                }
                if (lens.focusSurface >= 0)
                {
                    lensError(name, "surfaces %d and %d both have a focus gap; one is supported", lens.focusSurface + 1, (int)i + 1);
                }
                lens.focusSurface = (int)i;
                lens.focusGap = glm::vec2(f.at(0).get<float>(), f.at(1).get<float>());
                s.thickness = lens.focusGap.x;  // the table is at infinity
            }
            lens.surfaces.push_back(s);
        }
    }
    catch (const json::exception& e)
    {
        lensError(name, "%s", e.what());
    }
    validate(lens, name);
    return lens;
}

float lensLength(const LensSystem& lens)
{
    float z = 0.0f;
    for (const LensSurface& s : lens.surfaces)
    {
        z += s.thickness;
    }
    return z;
}
