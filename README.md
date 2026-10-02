CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Yichen Huang
    - [LinkedIn](https://www.linkedin.com/in/yichen-huang-970b582bb/), [personal website](https://as7tesia.com/)
* Tested on: Windows 11, AMD Ryzen 5950X @ 4.3GHz (PBO Enabled), 64GB(3200 MT/s), RTX 3090 24GB

### Status

What works today, and what each missing piece needs before it can work. Last updated 2026-10-01.

#### glTF

A `.gltf` or `.glb` file loads in two ways: as the whole scene when it is the scene argument, or as a `"TYPE":"mesh"` object inside a scene JSON.

| Part | Works today |
|---|---|
| Geometry | Triangle lists with float positions, indexed or not. Normals come from the file, or are generated area-weighted when it has none. Tangents come from the file, or are generated with MikkTSpace when the primitive's material has a normal map. Node transforms as a matrix or as translation, rotation, scale. One instance per (node, primitive) pair |
| Camera | The first perspective camera in the node tree gives position, direction and `yfov`. A camera transform that mirrors flips the image. A file without a camera gets one that frames the scene's bounding sphere |
| Lights | Emissive surfaces: `emissiveFactor` times the emissive texture times `KHR_materials_emissive_strength`. An emitter also reflects through its BSDF |
| Materials | The glTF metallic-roughness BSDF with GGX microfacets (see [Materials](#materials-gltf-metallic-roughness)): base color, metallic, roughness, normal map, alpha mask, `KHR_materials_ior`, `_transmission`, `_volume` absorption, and the factors of `_specular` and `_clearcoat` |
| Textures | Base color, metallic-roughness, normal, emissive and transmission. PNG and JPEG, as files next to the glTF, data URIs or buffer views, decoded by tinygltf through stb_image. 16-bit images are converted to 8 bits per channel. Texture coordinates from `TEXCOORD_0`, wrap modes and the magnification filter from the sampler, no mipmaps. The texture unit decodes sRGB (base color, emissive) to linear before filtering |
| Render settings | glTF has none. Resolution and depth come from the flags, then from `extras.pbrt.render` when the file has it, then from the defaults (1024 high, depth 8). 5000 spp unless `--spp` is given |

| glTF feature | What happens today | Needs |
|---|---|---|
| Alpha blend | Rendered opaque: every Bistro material (the foliage) and Intel Sponza's `dirt_decal` | Alpha as coverage in the any-hit program that already handles alpha mask: the ray passes with probability 1 - alpha, from a random number both intersection paths can compute |
| One-sided emitters (`doubleSided` false) | An emitter emits from both faces | A front-face test on emission in the shade kernel |
| Environment light (PBRT infinite light in the research scenes) | Ignored, named on stderr | Environment lighting: radiance returned when a ray misses, from an HDRI map (`.exr`, `.pfm`) or a constant color |
| Punctual lights (`KHR_lights_punctual`), PBRT distant lights | Ignored, named on stderr | Direct light sampling (next event estimation). These lights have no surface for a path to hit |
| Scenes lit through a small opening (veach-ajar) | Mostly noise | Direct light sampling (next event estimation) |
| Camera roll | Dropped, with a note printed | An interactive camera that keeps its own up vector. The viewport camera always uses world +Y |

Of the 26 [glTF research scenes](https://github.com/ErfanMo77/gltf-research-scenes), 15 have an emissive surface and render with their glTF materials. The other 11 render black: 10 are lit by an environment light, and dragon by a distant light alone. cornell-caustic renders nearly black too, as it did with every material diffuse: its `extras.pbrt.render` names SPPM as the integrator.

Skipped by the loader, with no work planned: occlusion maps (they stand in for the shadowing a path tracer computes), the textures of `KHR_materials_specular` and `_clearcoat` (the clearcoat normal map among them), the thickness of `KHR_materials_volume`, sheen and the other `KHR_materials_*` extensions, vertex colors, texture coordinate sets after `TEXCOORD_0`, `KHR_texture_transform`. The loader names the extensions it skips, on the materials and on the texture slots it reads, and counts the blend materials on stderr.

### Viewport controls

The viewport navigates like the Unreal Engine 5 level editor: hold the right mouse button to look around and fly, or hold Alt for the Maya-style orbit. Any camera move restarts the accumulation.

| Input | Action |
|---|---|
| RMB drag | Look around in place |
| RMB + W / S | Fly forward / back |
| RMB + A / D | Fly left / right |
| RMB + E / Q | Fly up / down along world +Y |
| RMB + wheel | Fly speed up / down, x1.25 per notch (shown in the ImGui panel) |
| LMB drag | Up / down moves along the ground, left / right turns |
| MMB drag, or LMB + RMB drag | Pan |
| Wheel | Move forward / back in steps |
| Alt + LMB drag | Orbit around the pivot |
| Alt + RMB drag | Move toward / away from the pivot |
| Alt + MMB drag | Pan |
| F | Back to the scene file's camera |
| Ctrl + S | Save the image |
| Esc | Save the image and quit |

The ImGui panel also switches scenes without a restart. Its combo lists the scene JSONs in `scenes/` and the catalog's names (entries whose glTF is not downloaded are grayed out), and the field under it takes anything the command line takes: a name or a path to a .json, .gltf or .glb. The new scene is read first, so a bad file leaves the current scene in place with the error in the panel. On success the device buffers are rebuilt, the window takes the new scene's resolution, the camera starts at its scene camera and the accumulation starts over; `--res`, `--spp` and `--depth` from the command line still apply. The window is busy while a large scene loads (Intel Sponza takes a few seconds).

The camera is a position, a yaw about world +Y, a pitch that stops just short of straight up or down, and a pivot distance. The orbit pivot sits on the view axis at that distance and travels with the camera. It starts at the scene's `LOOKAT`, or for a glTF camera at a point on its view axis near the middle of the scene. The default fly speed crosses the scene's bounding box diagonal in 4 seconds. A drag hides and locks the cursor, so it does not stop at the edge of the screen. Headless renders build their basis from the same pose as the window's first frame.

### Headless rendering
Render without a viewport
```
./build/bin/Release/cis565_path_tracer.exe scenes/cornell.json --headless --spp 100 --res 400x400 --out img/test/cornell.png
```

The scene argument is a path, or a name: `cornell` is `scenes/cornell.json`, and `veach-mis` is the research scene of that name. Names other than the scene JSONs come from `scenes/catalog.json`, which maps a short name to each glTF scene under `scenes/assets` (the 26 research scenes in their extended variant, and the two Bistro files), so the files stay where their downloads put them. `--list` prints every name and marks the ones not downloaded.

| Flag | Effect |
|---|---|
| `--headless` | No GLFW / ImGui / OpenGL. Render, save, exit. Prints total time and ms per sample. |
| `--list` | Print the scene names and exit |
| `--spp N` | Override `ITERATIONS` from the scene file |
| `--res WxH` | Override `RES` from the scene file |
| `--depth N` | Override `DEPTH` from the scene file, the most rays a path may trace |
| `--out path.png` | Write exactly this file. Default is the usual `img/auto_saved/<FILE>.<time>.<spp>samp.png` |
| `--no-rr` | Disable Russian roulette |
| `--sort` | Sort paths by material before shading. Off by default since the [profile](#the-material-sort); `--no-sort` is the default, kept for scripts |
| `--no-optix` | Intersect with the naive per-object kernel instead of the OptiX stage |
| `--optix-validate` | OptiX validation mode: checks every launch, slow |
| `--timing` | With `--headless`: print load, init and per-bounce stage times as CSV lines (see [Performance](#performance)) |
| `--tonemap none\|aces\|agx\|agx-punchy` | View transform for viewport and PNG. Default `agx-punchy`. `none` is the raw clamp the base code shipped with |
| `--exposure X` | Linear multiplier before the view transform. Default 1.0 |

The overrides also work in windowed mode. Output is deterministic, with same scene, spp, resolution and tonemap produce a byte-identical PNG, so `cmp` against a previous render can be used to prove correctness for things that only improves performance but shouldn't alter the image at the same sample count.

`tests/cmp_renders.sh` runs that check over 13 scenes (the JSON scenes, the textured and masked glTF ones, the three Khronos material tests and a research scene), both intersection paths, at 400x400 and 401x399. `golden` renders the set into `build/golden` before a change, `compare` renders it again after and cmps every PNG against its golden, and the exit code is nonzero when any of them differs.

### View transform

The accumulation buffer is scene-linear and never touched. Tonemapping is applied once at display and once at save, through the same function in `src/render/tonemap.h`, so the viewport and the PNG agree. Default is AgX punchy.

- **AgX** by Troy Sobotka, published as an OpenColorIO config: https://github.com/sobotka/AgX. The analytic version this project uses (inset/outset matrices, log2 shaper, polynomial sigmoid) is Benjamin Wrensch's "Minimal AgX Implementation": https://iolite-engine.com/blog_posts/minimal_agx_implementation. `agx-punchy` adds the "punchy" look from the same post, an ASC CDL grade (power 1.35, saturation 1.4) between the sigmoid and the outset matrix, matching the look of that name in Blender.
- **ACES**, two fits of the reference RRT+ODT, picked by the `ACES_FIT_HILL` macro in `tonemap.h`. Default is Krzysztof Narkowicz's single rational curve: https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/. The alternative is Stephen Hill's fit with the sRGB to AP1 round trip, from `ACES.hlsl` in MJP's BakingLab: https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl. Hill's is scaled by 1/0.6 on input, same as three.js, so the two match in brightness. On Cornell they are within a few levels of each other once that's done.

Same accumulation buffer through each transform. Glass Cornell, 600x600, 1000 spp, exposure 1.0, DEPTH 8:

<!-- TODO: replace these four with the final hero shot through the same four transforms. Cornell has no saturated highlights, so the hue-skew difference between ACES and AgX barely shows here. -->

| ACES, Narkowicz fit | ACES, Hill fit (1/0.6) |
|:---:|:---:|
| ![ACES Narkowicz](img/readme/tonemap_aces_narkowicz.png) | ![ACES Hill](img/readme/tonemap_aces_hill.png) |
| **AgX** | **AgX punchy** |
| ![AgX](img/readme/tonemap_agx.png) | ![AgX punchy](img/readme/tonemap_agx_punchy.png) |

### Meshes: glTF loading

glTF files load through [tinygltf](https://github.com/syoyo/tinygltf) (v2.9.7, the last C++ header; v3 is a C rewrite). A scene object of `"TYPE":"mesh"` names the file, relative to the scene file's folder:

```json
{
    "TYPE":"mesh",
    "FILE":"assets/Duck/glTF/Duck.gltf",
    "MATERIAL":"glass",
    "TRANS":[0.0,-0.3,0.0], "ROTAT":[0.0,-25.0,0.0], "SCALE":[3.0,3.0,3.0]
}
```

The loader walks the file's default scene, multiplies node transforms down the tree, and makes one instance per (node, primitive) pair: the object's TRANS/ROTAT/SCALE times the node's world matrix. The positions, normals, texture coordinates, tangents and triangle indices of every primitive go into one flat array each on the `Scene`, read through the accessor's buffer view and stride, with a per-primitive record of where its triangles start. A primitive without normals gets area-weighted vertex normals, one without `TEXCOORD_0` gets (0, 0) at every vertex. A primitive without `TANGENT` whose material has a normal map gets tangents from [MikkTSpace](https://github.com/mmikk/MikkTSpace) (Morten S. Mikkelsen, zlib license, vendored in `external/mikktspace/`), the generator glTF names and the one normal maps are baked against; every other primitive gets zero tangents. `MATERIAL` is optional: without it the file's materials are appended to the scene's material list (see [Materials](#materials-gltf-metallic-roughness)).

Both intersection paths read the same arrays. The naive kernel loops over every triangle of every mesh instance in object space (two-sided Moller-Trumbore). OptiX builds one triangle GAS per primitive over the same buffers and one instance per scene object; its closest-hit program finds the triangle through a per-instance record and interpolates the vertex normal, uv and tangent from the barycentrics. Same conventions on both sides: the geometric normal decides `outside`, the shading normal is flipped to face the ray on a back-face hit, so refraction through a mesh works the same as through the built-in sphere.

Each image a material uses becomes a CUDA array of `uchar4`, and each (image, sampler, color space) a texture object over it (`src/render/textures.cpp`). The shade kernel samples each of a material's textures with `tex2D` at the hit's uv and multiplies the matching factor by it. Base color and emissive are stored sRGB-encoded, and the texture object's `sRGB` flag has the texture unit decode each texel to linear before the bilinear blend, so memory keeps the 8-bit sRGB values and the shader gets linear floats. A 16-bit image is rounded to 8 bits per channel at load. There are no mipmaps: the camera jitters each pixel's ray, so the samples already average the texels a pixel covers.

#### Test assets

Models come from [KhronosGroup/glTF-Sample-Assets](https://github.com/KhronosGroup/glTF-Sample-Assets), folder `Models/<name>/glTF/`, and are expected under `scenes/assets/<name>/glTF/`. The three material tests are the `.glb` files from `Models/<name>/glTF-Binary/`, under `scenes/assets/KhronosTests/`; they are made for environment lighting and have no light of their own, so their scene files add an emitting panel. That folder is gitignored. Each model needs its `.gltf`, `.bin` and the image files next to them. An image that is missing or cannot be used (a file that is not there, an image that only a KTX2 or WebP extension names) is a warning on stderr, and a base color slot that uses it renders magenta, so the broken asset shows in the render. The other slots fall back to their factor, since magenta in them would mean a mirror (metallic-roughness), a sideways normal or a surface that glows.

| model | triangles | scene file | credit |
|---|---|---|---|
| Box | 12 | `cornell_boxmesh.json` | Cesium, 2017, CC BY 4.0 |
| Duck | 4,212 | `cornell_duck.json` | Sony, 2006, SCEA Shared Source License 1.0 |
| Suzanne | 3,936 | `cornell_suzanne.json` | Norbert Nopper / UX3D, 2017, CC0 |
| DamagedHelmet | 15,452 | `cornell_helmet.json` | theblueturtle_, 2016, CC BY-NC 4.0; glTF rebuild by ctxwing, 2018, CC BY 4.0 |
| Sponza | 262,267 | `sponza.json` | Crytek (Frank Meinl), CryENGINE Limited License Agreement, glTF conversion from the Khronos repo |
| Intel Sponza, main and curtains | 3,747,018 + 1,997,366 | `sponza_intel.json` | Frank Meinl and Anton Kaplanyan, Intel, 2022, CC BY 4.0; curtains add-on by the Sponza Addon Package Crew |
| Bistro exterior | 2,829,226 | `assets/Bistro/Exterior/BistroExterior.gltf`, loaded directly | Amazon Lumberyard, July 2017, CC BY 4.0 |
| Bistro interior | 1,043,077 | `assets/Bistro/Interior/BistroInterior.gltf`, loaded directly | Amazon Lumberyard, July 2017, CC BY 4.0 |
| TransmissionRoughnessTest | 77,792 | `transmission_roughness.json` | Ed Mackey, Analytical Graphics, 2021, CC BY 4.0 |
| DragonAttenuation | 134,995 | `dragon_attenuation.json` | Dragon: Stanford Computer Graphics Laboratory, 1996, converted by Morgan McGuire, 2017, Stanford Graphics Library license; cloth backdrop: Adobe, CC0 |
| ClearCoatTest | 37,116 | `clearcoat.json` | Ed Mackey, Analytical Graphics, 2020, CC BY 4.0 |

The Intel Sponza comes from the [Intel Sample Library](https://www.intel.com/content/www/us/en/developer/topic-technology/graphics-processing-research/samples.html) as one zip per package. The glTF files, their `.bin` and `textures/` folders go under `scenes/assets/IntelSponza/<package>/`, keeping the zip's folder names (`main_sponza`, `pkg_a_curtains`, `pkg_d_10k_candles`). `sponza_intel.json` loads the main scene and the curtains package together, with the main file's first camera (`PhysCamera001`) and an emitting slab above the atrium in place of the sun, which is a punctual light. Cited as: Frank Meinl and Anton Kaplanyan, 2022, Intel Sample Library. The add-on packages also credit the Sponza Addon Package Crew: Katica Putica, Cristiano Siqueira, Timothy Heath, Justin Prazen, Sebastian Herholz, Bruce Cherniak, Anton Kaplanyan.

The Bistro comes from NVIDIA's Open Research Content Archive as FBX with DDS textures (`Bistro_v5_2.zip`), cited as: Amazon Lumberyard Bistro, Open Research Content Archive (ORCA). Amazon Lumberyard, July 2017. http://developer.nvidia.com/orca/amazon-lumberyard-bistro. I converted it with Blender 5.2.2: FBX import, then a glTF Separate export into `scenes/assets/Bistro/Exterior/` and `scenes/assets/Bistro/Interior/`, which writes every DDS texture out as PNG. Bistro's "Specular" texture packs occlusion, roughness and metalness into R, G and B, and the FBX importer wires it to Specular IOR Level, which exports as `KHR_materials_specular`. In the saved `.blend` files a Separate Color node sends G to Roughness and B to Metallic instead, so the export writes the same image as glTF's `metallicRoughnessTexture` (glTF reads roughness from G and metalness from B too). Every Bistro emitter has an emissive texture, and the scenes are otherwise lit by punctual lights and the sun, so both render dark until direct light sampling. Bistro's normal maps follow the DirectX convention (its README lists them as "Normal (DirectX)"), whose green channel points the opposite way from glTF's. They are loaded unchanged, so detail along the texture's v direction is lit inverted.

#### Naive loop vs OptiX by triangle count

1024x1024, ms per sample, median of 3 interleaved runs, GPU otherwise idle, from the [profiling sweep](#performance). `--no-optix` runs the naive kernel; the naive runs use 1 to 4 samples each because of how long they take.

| scene | triangles | depth | naive | OptiX |
|---|---|---|---|---|
| cornell (spheres and cubes, no mesh) | 0 | 8 | 8.41 | 8.63 |
| cornell-box (research scene) | 36 | 66 | 11.56 | 11.43 |
| cornell_helmet | 15,452 | 8 | 320 | 8.42 |
| transmission_roughness | 77,792 | 12 | 1,511 | 7.37 |
| dragon_attenuation | 134,995 | 16 | 3,146 | 7.52 |
| sponza | 262,267 | 8 | 5,553 | 10.20 |

![Naive loop vs OptiX by triangle count](img/readme/perf_naive_vs_optix_1024x1024.png)

The naive kernel's time grows linearly with the triangle count, about 20 ms per sample per thousand triangles at this resolution, because every path tests every triangle of every instance at every bounce. OptiX stays within 15% of the Cornell frame time up to Sponza; Intel Sponza (5.7 million triangles) and Bistro (2.8 million) take 12.7 and 13.8 ms on OptiX and were not run on the naive kernel. The two analytic scenes come out even on both paths: with seven objects the naive loop is as cheap as a launch.

### Materials: glTF metallic-roughness

Every material, from a glTF file or a scene JSON, is one material model: the metallic-roughness BSDF from Appendix B of the glTF 2.0 spec, plus five `KHR_materials_*` extensions. The scene JSON format keeps its four material types, and the loader translates each into parameters of that model, starting from glTF's default material:

| JSON `TYPE` | Parameters |
|---|---|
| `Diffuse` | base color `RGB`, metallic 0, `specularFactor` 0: pure Lambert |
| `Emitting` | emission `RGB` x `EMITTANCE` over a black base with metallic 0 and `specularFactor` 0, which reflects nothing, so the path ends there |
| `Specular` | base color `RGB`, metallic 1, roughness `ROUGHNESS` (optional, default 0: a perfect mirror) |
| `Refractive` | base color `RGB`, transmission 1, ior `IOR`, roughness 0 |

The JSON types used to have their own scatter function, and two of them render differently now: the mirror brightens toward white at grazing angles (Fresnel with the base color as f0, instead of a flat tint), and glass tints only the refracted light, not the reflection. Every mirror and glass in the scene files is white or 0.98 white, so on them the change is mostly the noise pattern.

```
coated     = fresnel_coat(material, GGX(clearcoatRoughness^2))        KHR_materials_clearcoat
material   = mix(dielectric, metal, metallic)
metal      = GGX(roughness^2) with Schlick Fresnel, f0 = base color
dielectric = fresnel_mix(layer = GGX(roughness^2),
                         base  = mix(Lambert(base color), GGX transmission x base color, transmission))
```

The dielectric's f0 comes from `KHR_materials_ior` (1.5 by default, f0 = 0.04), times `specularColorFactor` and weighted by `specularFactor` from `KHR_materials_specular`. Per hit, `src/render/pbr_surface.h` reads the base color, metallic-roughness (roughness in G, metallic in B), normal, emissive and transmission (R) textures and multiplies each factor by its texture. The base color's alpha is read by the alpha test in the intersection stage instead.

#### Sampling

`scatterPbr` (`src/render/bsdf.cu`) picks one lobe per bounce and samples a direction from it. Microfacet normals come from the GGX distribution of normals visible from the outgoing direction (Dupuy and Benyoub 2023, spherical caps), so the D term and the Jacobian of the reflection or refraction cancel against the pdf, and a microfacet lobe's weight is its Fresnel term times the height-correlated Smith G2/G1. Roughness below 0.01 is a perfect mirror or a clean refraction.

| choice | picked with probability | sample weight |
|---|---|---|
| clearcoat or the rest | clearcoat x Fresnel(0.04, n·v) | G2/G1 |
| metal or dielectric | metallic | Fresnel(base color, v·h) x G2/G1 |
| glass or opaque dielectric | transmission | the chosen lobe's |
| glass: reflect or refract | Fresnel at the sampled microfacet | G2/G1, and x base color for refraction |
| opaque: specular layer or Lambert, p = layer / (layer + base) | layer: specularFactor x max Fresnel(n·v); base: max of the base color | layer: specularFactor x Fresnel(v·h) x G2/G1 / p; Lambert: base color x (1 - specularFactor x max Fresnel(v·h)) / (1 - p) |

Where the probability is the mixing weight itself, the weight cancels. The last row cannot do that, because the layer's Fresnel term depends on the sampled microfacet, so it picks with an estimate made from n·v and divides by it. The first version estimated the base as (1 - layer) x base color, which goes to 0 at grazing angles while the Fresnel term at a cosine-sampled half vector stays small, and the rare diffuse samples there got weights near 100. With the base color alone a diffuse weight stays below 2. A sample that ends on the wrong side of the surface (a normal map or a rough lobe can produce one) ends the path. The coat lobe exists on the air side only: inside a transmissive material (a back-face hit) the coat and the glass share an index, so the path sees no coat.

#### Transmission and absorption

Transmission is a solid boundary: the ray refracts through the ior on the way in and on the way out, with or without `KHR_materials_volume`. glTF reads a transmissive material without the volume extension as thin-walled, with no bending. I chose solid because 31 of the 34 glass materials in the research scenes have no volume extension and were converted from PBRT's solid dielectrics, and Blender exports glass without it. The volume extension adds absorption: a path that refracts into a surface records that material as the medium it is in, and until it refracts out again the shade kernel multiplies its throughput by exp(-σt) at every hit, whatever the hit lands on, with σ = -ln(attenuationColor) / attenuationDistance and t the length of the segment that just ended. The path keeps one medium: refracting out of an object nested inside another counts as leaving both. The base color tints every crossing of the boundary.

#### Emission

Emission is `emissiveFactor` x emissive texture x `emissiveStrength`. The shade kernel adds throughput x emission to the pixel at every hit and the path goes on, so an emitter also reflects. This is the only place light reaches the image: a path that ends (a miss, its last bounce, Russian roulette, a sample that carries no light) adds nothing. The clearcoat sits above the emission and darkens it by 1 - clearcoat x Fresnel(0.04, n·v), on the air side only like the coat lobe.

#### Normal maps

The tangent frame is the interpolated tangent made perpendicular to the normal, and the bitangent w x cross(N, T). It is built on the front-face normal, the side the map was authored for, and the mapped normal is flipped for a back-face hit. A mapped normal that faces away from the ray (a steep map at a grazing angle) falls back to the unmapped one. A mirroring instance transform flips w. The clearcoat lobe keeps the unmapped normal.

MikkTSpace takes v pointing up the image and glTF's v points down, so the loader hands it 1 - v. In Khronos Sponza, which ships its own `TANGENT`, 99.97% of the triangles have the bitangent pointing toward decreasing v, the direction this produces. Where MikkTSpace gives the corners of a shared vertex different tangents (a mirrored uv seam), the loader copies the vertex.

#### Alpha mask

A hit on a `MASK` material whose alpha (base color factor alpha x texture alpha) is below `alphaCutoff` does not count, and the ray goes on. OptiX runs an any-hit program for this on masked instances only: a mesh GAS keeps any-hit when one of its instances is masked, and the instance flag turns it off on the rest. The naive kernel makes the same test on every hit that would become the closest. The texture unit reads alpha linearly even from an sRGB texture (a texel of 128 reads 0.502 with the sRGB flag on, while its color reads 0.216). On a Sponza plant close-up the two paths differ by 0.15/255 on average.

#### Checking the BSDF

A host-side test (`tests/bsdf_test.cu`, the `bsdf_test` CMake target, outside the default build) runs `scatterPbr` two million times per case and compares the mean weight, which is the directional albedo, to a brute-force integral of the analytic BSDF over the sphere. Roughness 0.6, view angle 0.8 rad from the normal, red channel:

| material | sampled | integral |
|---|---|---|
| metal, base color (1, 0.7, 0.3) | 0.7941 | 0.7948 |
| opaque dielectric, base color (0.5, 0.8, 0.2) | 0.5147 | 0.5146 |
| the same with `specularFactor` 0 | 0.5000 | 0.4999 |
| glass, ior 1.5, white | 0.9615 | 0.9611 |
| metallic 0.4, transmission 0.3, `specularFactor` 0.7 | 0.8208 | 0.8210 |
| clearcoat 1 (roughness 0.3) over red plastic | 0.5347 | 0.5346 |

All 54 cases (six materials, roughness 0.3, 0.6 and 1, three view angles) agree within the noise of the integral. The test takes each estimate's standard error from its own samples and fails a case whose difference is more than 5 combined standard errors (the highest here is 2.1). The largest gaps, up to 0.008 for glass and metal at roughness 0.3, belong to the integral: with 16 times the samples it moves by as much (glass at 0.8 rad goes from 0.9895 to 0.9999, against 0.9975 sampled), since uniformly drawn directions rarely land in a narrow lobe. Two million random inputs, including grazing views and tilted shading normals, produce no NaN, infinite or negative weight. In the smooth limit the checks are exact: smooth metal returns the Schlick term in the mirror direction, and smooth white glass weighs every sample 1 and reflects the Fresnel share, all of it past the critical angle. The exit code is nonzero when any of the 58 checks fails.

The three Khronos material tests ship without lights, so their scene files add an emitting panel. 1024 spp, AgX.

| TransmissionRoughnessTest | DragonAttenuation |
|:---:|:---:|
| ![TransmissionRoughnessTest](img/readme/materials_transmission_roughness.png) | ![DragonAttenuation](img/readme/materials_dragon_attenuation.png) |
| **ClearCoatTest** | **DamagedHelmet** |
| ![ClearCoatTest](img/readme/materials_clearcoat.png) | ![DamagedHelmet](img/readme/materials_cornell_helmet.png) |

- **TransmissionRoughnessTest**: rows from ior 1.0 to 2.42, roughness growing to the right. The ior 1.0 row stays sharp at every roughness, since a microfacet cannot bend a ray that crosses no change in index, and the blur grows with the ior.
- **DragonAttenuation**: one absorption color and distance; the thin claws and spines come out pale yellow and the thick body deep orange.
- **ClearCoatTest**: the middle column combines the base layer's broad highlight (left) with the coating's sharp one (right). Four of its rows depend on clearcoat textures, which are not read, so their coat is uniform.
- **DamagedHelmet**: normal map (tangents from MikkTSpace), metallic-roughness map and the emissive lamps.

### Performance

A profile of the whole renderer, from the scene file to the saved image, taken on 2026-10-02 at commit c2e44f5 (after the glTF materials and the code cleanup). RTX 3090, driver 616.56, CUDA 13.3, Windows 11 (WDDM), Release build with device LTO. Every number is the median of 3 interleaved runs with the GPU otherwise idle (nvidia-smi under 10% before each run), at 1024x1024 unless stated.

**How it was measured.** `--timing` wraps every stage of every bounce in cudaEvents and prints the averages over the samples as CSV, next to host timers around the loader, the uploads and the OptiX builds. A stage's time is what the GPU timeline shows between its two events: its kernels, the launch gaps, and any host synchronization inside it. The flag changes nothing when it is off and nothing in the image when it is on (`tests/cmp_renders.sh`, 52 of 52 identical). `profiling/run_sweep.py` runs the matrix below and `profiling/analyze.py` makes the tables and charts; the raw records are in `profiling/results/sweep.jsonl`. Nsight Systems 2026.4.1 gave the per-kernel times and the host-side API picture, `cuobjdump --dump-resource-usage` the registers, and Nsight Compute the occupancy.

| scene | why it is in the set | triangles | materials and textures |
|---|---|---|---|
| cornell | the base scene, analytic shapes, JSON materials | 0 | 5, none |
| cornell-box | the research scene: matte PBR, no textures, depth 66 from its file | 36 | 7, none |
| cornell_helmet | every texture slot and a normal map | 15,452 | 6, 5 |
| transmission_roughness | glass, depth 12 | 77,792 | 9 |
| dragon_attenuation | an open scene with a glass mesh, depth 16 | 134,995 | 3 |
| sponza | alpha mask | 262,267 | 25, 69 |
| sponza_intel | millions of triangles, 4K textures, alpha blend | 5,744,384 | 405 instances, 80 images |
| bistro-exterior | millions of triangles, open sky, every material alpha blend | 2,829,226 | 1,591 instances, 405 images |

#### Where a sample goes

![Stage breakdown](img/readme/perf_stage_breakdown_1024x1024.png)

| scene | generate | intersect | sort | shade | compact | ms per sample |
|---|---|---|---|---|---|---|
| cornell | 0.20 | 0.91 | 6.16 | 0.71 | 0.67 | 8.63 |
| cornell-box | 0.21 | 1.70 | 7.34 | 0.85 | 1.26 | 11.43 |
| cornell_helmet | 0.20 | 0.89 | 5.86 | 0.82 | 0.65 | 8.42 |
| transmission_roughness | 0.21 | 1.17 | 4.57 | 0.68 | 0.75 | 7.37 |
| dragon_attenuation | 0.20 | 1.59 | 4.22 | 0.66 | 0.84 | 7.52 |
| sponza | 0.21 | 1.70 | 6.46 | 1.20 | 0.63 | 10.20 |
| sponza_intel | 0.20 | 3.69 | 6.45 | 1.66 | 0.65 | 12.68 |
| bistro-exterior | 0.20 | 4.67 | 7.04 | 1.19 | 0.67 | 13.81 |

- **The material sort is the largest stage in every scene**, 45% of a sample on Bistro and 71% on cornell. It is also the stage the scene has the least say in: 4.2 to 7.3 ms whether the scene is seven analytic objects or three million triangles.
- **Intersection is the second**, and the only stage that follows the scene: 0.9 ms on cornell, 1.7 ms on Sponza, 3.7 ms on Intel Sponza, 4.7 ms on Bistro. RT cores keep a 5.7 million triangle scene at 4x the cost of seven spheres and cubes.
- **Shading is 0.7 to 1.7 ms**, 6 to 13% of a sample, with the full glTF BSDF and every texture slot. The step 3 materials are not where the time went.
- **Compaction is 0.6 to 1.3 ms** and generation 0.2 ms.

The same breakdown at 400x400 (160 thousand paths) sums to 2.1 to 3.7 ms per sample, of which the sort is still 1.0 to 1.4 ms. Below about 20 thousand live paths every stage sits on a floor of 10 to 50 us per bounce, which is launch latency and the synchronizations rather than work; see [Host synchronization](#host-synchronization). That floor is why the cost per sample is not proportional to the pixel count: from 801x799 to 1024x1024 it scales with the pixels, while 400x400 costs 1.5 to 2x what that line predicts.

![ms per sample against resolution](img/readme/perf_resolution_scaling.png)

**Per bounce.** The stage mix changes with the depth. Intel Sponza at 1024x1024:

![Per-bounce stages, Intel Sponza](img/readme/perf_bounces_sponza_intel_1024x1024.png)

| bounce | paths in | intersect | sort | shade | compact |
|---|---|---|---|---|---|
| 1 | 1,048,576 | 0.61 | 1.39 | 0.58 | 0.20 |
| 2 | 882,140 | 1.30 | 2.64 | 0.54 | 0.17 |
| 3 | 721,600 | 1.25 | 2.15 | 0.45 | 0.08 |
| 4 | 25,494 | 0.14 | 0.09 | 0.03 | 0.03 |
| 5 to 8 | 5,042 and fewer | 0.08 to 0.11 | 0.04 to 0.07 | 0.01 to 0.02 | 0.04 |

The first bounce's intersection is cheap because the camera rays are coherent; bounces 2 and 3 cost twice as much for fewer paths once the rays scatter. The sort costs twice as much at bounce 2 as at bounce 1 with fewer paths, for a reason given below. From bounce 4 on the work is small and the bounce costs its fixed overhead.

**Paths alive.** Russian roulette starts at the third bounce with a survival probability equal to the throughput's luminance, so in a scene whose albedo products are small most paths end there:

![Paths alive after each bounce](img/readme/perf_alive_paths_1024x1024.png)

| scene | after bounce 1 | 2 | 3 | 4 | 8 |
|---|---|---|---|---|---|
| cornell (closed, bright walls) | 82% | 57% | 28% | 18% | 5.5% |
| cornell-box (closed, depth 66) | 99.5% | 74% | 20% | 9.7% | 0.7% |
| sponza (closed, textured) | 93% | 85% | 4.0% | 0.8% | 0.0% |
| sponza_intel | 84% | 69% | 2.4% | 0.5% | 0.0% |
| bistro-exterior (open) | 92% | 75% | 4.7% | 1.2% | 0.0% |
| dragon_attenuation (open) | 90% | 41% | 9.2% | 4.0% | 0.4% |

In the open scenes the losses at bounces 1 and 2 are misses (there is no environment light yet, a miss ends the path); in the textured scenes the drop at bounce 3 is roulette, since two bounces off dark stone leave a throughput near 0.1. Both make the loop's later bounces cheap and the first three the whole cost.

#### The material sort

`--no-sort` leaves the paths in their compacted order and shades them as they come.

![Material sort on and off](img/readme/perf_sort_toggle_1024x1024.png)

| scene | sort on | sort off | sort costs | shade with sort | shade without |
|---|---|---|---|---|---|
| cornell | 8.63 | 2.47 | 3.49x | 0.70 | 0.75 |
| cornell-box | 11.43 | 3.91 | 2.92x | 0.87 | 0.97 |
| cornell_helmet | 8.42 | 2.46 | 3.42x | 0.82 | 0.82 |
| transmission_roughness | 7.37 | 2.69 | 2.74x | 0.68 | 0.71 |
| dragon_attenuation | 7.52 | 3.23 | 2.33x | 0.66 | 0.71 |
| sponza | 10.20 | 4.11 | 2.48x | 1.20 | 1.52 |
| sponza_intel | 12.68 | 6.40 | 1.98x | 1.66 | 1.71 |
| bistro-exterior | 13.81 | 6.77 | 2.04x | 1.19 | 1.30 |

The sort was kept after the CUB rewrite on the expectation that it would pay off once shading got expensive. The profile says the opposite on every scene, and says why:

- **The gather is the cost, not the sort.** Nsight Systems on cornell (50 samples): `gatherByIndex` 284 ms over 400 launches, 711 us each, against 34 ms for `shadeMaterial`, 31 ms for the OptiX launches and 26 ms for compaction. The radix kernels themselves are 14, 4 and 3 us per launch. The gather moves two 48-byte records per path (96 MB per bounce at a million paths) through a permutation, and at bounce 2 the permutation is random: bounce 1's gather reads paths that are still in pixel order and runs at 1.1 ms, bounce 2's reads the already permuted buffers and takes 2.0 to 3.0 ms for fewer paths.
- **Shading gains at most 0.3 ms from it.** The last two columns: sorting by material makes `shadeMaterial` 0.05 ms faster on cornell and 0.32 ms faster on Sponza, the scene with the most materials, against a sort stage of 6 ms. The kernel is short enough, and the materials alike enough (one BSDF, lobes picked per thread), that a warp with mixed materials costs about what a sorted one does.
- **The 400x400 numbers from September were the same story**: 1.49 ms per sample with the sort against 0.68 without, read then as "the sort loses 2.2x on Cornell, wait for PBR". With PBR it loses 2.0x on Intel Sponza.

So the default is now off. `--sort` turns it on. A sort that would pay would have to stop moving the records: sort keys and indices only and let the shade kernel read through the index, or shrink the record by deferring the attribute interpolation to shading. Neither is worth doing while shading is 10% of the frame.

#### Russian roulette

`--no-rr` traces every path to the scene's depth.

![Russian roulette on and off](img/readme/perf_rr_toggle_1024x1024.png)

| scene | depth | RR on | RR off | RR saves |
|---|---|---|---|---|
| cornell | 8 | 8.63 | 11.04 | 1.28x |
| cornell-box | 66 | 11.43 | 30.14 | 2.64x |
| cornell_helmet | 8 | 8.42 | 11.63 | 1.38x |
| transmission_roughness | 12 | 7.37 | 10.36 | 1.41x |
| dragon_attenuation | 16 | 7.52 | 10.98 | 1.46x |
| sponza | 8 | 10.20 | 23.10 | 2.26x |
| sponza_intel | 8 | 12.68 | 23.64 | 1.86x |
| bistro-exterior | 8 | 13.81 | 25.61 | 1.85x |

The saving follows the alive-paths table: where roulette ends 95% of Sponza's paths at bounce 3, the loop without it carries them all to bounce 8 at full cost. Those are also the paths that carry the least light, which is what makes the estimator stay unbiased while the variance per sample grows a little.

#### Host synchronization

At the baseline the loop stopped for the host three times per bounce: `checkCUDAError` after the intersection launch (a `cudaDeviceSynchronize` when `ERRORCHECK` is 1), a second explicit `cudaDeviceSynchronize` right after it, and the readback of the alive count in `compactPaths`, which the host needs to size the next bounce's launches. Nsight Systems on cornell, 50 samples: 904 `cudaDeviceSynchronize` calls, 46 ms in total, 51 us each; and 800 `cudaMemcpyAsync` calls (the readback and the OptiX launch parameters) that block until the queued kernels finish, since the destination is pageable memory. Without the sort the GPU runs kernels for 60% of the render span; the other 40% is the host: each wait drains the queue, and the next kernel starts only after the host has noticed and launched it, 15 to 20 us later on WDDM.

`-DERRORCHECK=0` removes the first two waits (the `nosync` build in `profiling/results/sweep.jsonl`). Sort off, ms per sample:

| scene | 400x400, 3 waits | 1 wait | gain | 1024x1024, 3 waits | 1 wait | gain |
|---|---|---|---|---|---|---|
| cornell | 1.04 | 0.92 | 1.13x | 2.47 | 2.32 | 1.06x |
| cornell-box | 2.39 | 1.94 | 1.23x | 3.91 | 3.58 | 1.09x |
| cornell_helmet | 1.03 | 0.91 | 1.13x | 2.46 | 2.41 | 1.02x |
| transmission_roughness | 1.33 | 1.27 | 1.05x | 2.69 | 2.55 | 1.05x |
| dragon_attenuation | 1.77 | 1.70 | 1.04x | 3.23 | 3.05 | 1.06x |
| sponza | 1.43 | 1.33 | 1.08x | 4.11 | 3.91 | 1.05x |
| sponza_intel | 2.12 | 1.80 | 1.18x | 6.40 | 6.12 | 1.05x |
| bistro-exterior | 2.43 | 2.13 | 1.14x | 6.77 | 6.58 | 1.03x |

The saving is a fixed 15 to 20 us per bounce, which is why the research cornell-box with its 66-bounce depth (most of them nearly empty) gains the most and the million-pixel renders the least. `ERRORCHECK` is now 0 in Release builds; a kernel error still ends the run in the same bounce, at compaction's stream wait, with that call's name instead of the stage's. The Debug build keeps the per-stage checks.

The readback stays. It is the one wait the wavefront loop cannot avoid without restructuring (the K-bounce raygen loop idea, where the OptiX raygen program traces and shades the first bounces itself). Its ceiling is the per-bounce floor of about 0.1 ms, which is up to 30% of a 400x400 sample and 3% of a 1024x1024 one.

#### Kernel resources

`cuobjdump --dump-resource-usage` on the Release binary (sm_86, device LTO), and Nsight Compute on cornell_helmet and sponza at 1024x1024 for what the kernels reach at run time:

| kernel | registers | shared | local | notes |
|---|---|---|---|---|
| generateRayFromCamera | 20 | 0 | 0 | |
| computeIntersections (naive) | 80 | 0 | 0 | the per-object loop with the alpha test inlined |
| shadeMaterial | 64 | 0 | 0 | 42 before the glTF materials; the limit for 8 blocks of 128 threads per SM |
| gatherByIndex | 26 | 0 | 0 | |
| CUB DeviceSelectSweepKernel | 40 | 12 KB | 0 | compaction |
| CUB DeviceRadixSortOnesweepKernel | 80 | 33 KB | 0 | one pass over the material key's bits |
| CUB DeviceRadixSortHistogramKernel | 38 | 4 KB | 0 | |
| sendImageToPBO | 22 | 0 | 0 | viewport only |

No kernel spills to local memory. The OptiX programs are compiled by the driver from OptiX-IR and do not appear in the binary; Nsight Compute reports their launch as one kernel, below.

Nsight Compute over 45 consecutive launches from the third sample (cornell_helmet / sponza, sort on so the gather is in the set):

| kernel | registers | occupancy limit | achieved | SM throughput | DRAM throughput | warps stalled on a load |
|---|---|---|---|---|---|---|
| OptiX launch (raygen, hit and miss programs) | 78 | | 42% / 29% | 19% / 8% | 43% / 27% | 50% / 54% |
| shadeMaterial | 64 | 67% | 55% / 39% | 11% / 9% | 59% / 48% | 41% / 48% |
| gatherByIndex | 26 | 100% | 87% / 58% | 1% / 1% | 66% / 37% | 66% / 67% |
| DeviceSelectSweepKernel | 40 | 58% | 53% / 37% | 10% / 8% | 45% / 36% | 12% / 15% |
| generateRayFromCamera | 20 | 67% | 45% | 4% | 27% | 7% |
| DeviceRadixSortOnesweepKernel | 80 | 50% | 26% / 35% | 16% / 26% | 25% / 58% | 16% / 19% |

Every kernel that matters is memory-bound with the SMs mostly idle: the shade kernel at 11% of SM throughput against 59% of DRAM, the OptiX launch at 19%, the gather at 1% with two thirds of its warps waiting on a load. The registers are not what limits them. shadeMaterial's 64 registers allow 67% occupancy and it reaches 55% on the helmet scene; what it waits for is the 48-byte records and the texture fetches. So the step 3 cost (42 to 64 registers in the shade kernel) is not in the shade kernel: at 400x400 cornell went from 0.68 to 1.04 ms per sample sort off between 2026-09-25 and this commit, and from 1.49 to 2.06 sort on, with the extra 0.21 ms of the second pair being the gather of the record that grew from 32 to 48 bytes. A probe build that caps shadeMaterial at 40 registers through `__launch_bounds__(128, 12)` (88 bytes of stack spills, theoretical occupancy 100%) makes the shade stage 1.5 to 1.7x slower on every scene, 0.70 to 1.17 ms on cornell and 1.67 to 2.34 ms on Intel Sponza: the registers buy more than the occupancy would.

One cheap target shows in the table: generateRayFromCamera writes a million 48-byte records in 0.2 ms at 27% of DRAM peak, from 8x8 blocks storing the struct field by field. Now that the sort is gone it is 9% of a cornell sample.

#### Per-hit work

Two probe builds, each one change in a scratch copy of the source, for timing only (both change the image): one where `__closesthit__mesh` writes zero uv and tangent instead of interpolating them, and one where no instance is alpha tested, so the any-hit program never runs. 1024x1024, sort on like the baseline, ms per sample:

| scene | intersect | without uv and tangent | without any-hit | sample | without uv and tangent | without any-hit |
|---|---|---|---|---|---|---|
| cornell_helmet | 0.89 | 0.92 | | 8.42 | 8.64 | |
| transmission_roughness | 1.17 | 1.18 | | 7.37 | 7.82 | |
| dragon_attenuation | 1.58 | 1.43 | | 7.52 | 7.38 | |
| sponza | 1.70 | 1.62 | 1.52 | 10.20 | 9.58 | 10.04 |
| sponza_intel | 3.69 | 3.36 | 2.91 | 12.68 | 11.87 | 10.79 |
| bistro-exterior | 4.67 | 4.45 | 3.08 | 13.81 | 13.18 | 12.18 |

- **Interpolating uv and tangent at the hit costs at most 0.3 ms**, 9% of Intel Sponza's intersection stage and nothing measurable on the helmet. That bounds the plan's first two candidate fixes (per-instance flags to skip the attributes, or deferring the interpolation to shading): a few percent, and the smaller record they would also bring no longer matters with the sort off.
- **The texture fetches are about half of shading on the textured scenes.** The same probe's shade stage fell from 1.20 to 0.64 ms on Sponza, 1.67 to 0.62 on Intel Sponza and 1.19 to 0.64 on Bistro, because a uv of zero sends every fetch to one texel.
- **The any-hit program costs 0.2 ms on Sponza, 0.8 on Intel Sponza and 1.6 on Bistro**, 10 to 34% of the intersection stage. Sponza has three masked materials; Bistro has every material marked `BLEND`, so any-hit runs on every hit of every instance, fetches the alpha and hashes a random number, for materials whose alpha is 1 nearly everywhere. The next cheap fix is at load: a `BLEND` material whose alpha factor is 1 and whose base color texture holds no texel below 1 is opaque and gets its any-hit disabled; it changes nothing in the image.
- **The random engine was not probed.** thrust's default engine is a seed hash and a few multiplies per number, and the Nsight Compute table above has shadeMaterial at 11% SM throughput: whatever ALU work it does is hidden behind the memory.

#### Load and init

Host timers around the loader and `pathtraceInit`, same runs. The large scenes:

![Load and init breakdown](img/readme/perf_load_breakdown.png)

| scene | glTF parse | image decode | nodes and vertices | MikkTSpace | load total | texture upload | OptiX builds | init total | GPU memory |
|---|---|---|---|---|---|---|---|---|---|
| cornell | 0 | 0 | 0 | 0 | 0.002 s | 0 | 0.012 s | 0.20 s | 516 MB |
| cornell_helmet | 0.006 | 0.13 | 0.01 | 0.01 | 0.16 s | 0.008 | 0.013 | 0.21 s | 582 MB |
| sponza | 0.08 | 0.69 | 0.04 | 0 | 0.83 s | 0.04 | 0.04 | 0.28 s | 824 MB |
| sponza_intel | 2.3 | 27.6 | 0.8 | 0 | 31.0 s | 0.61 | 0.19 | 1.06 s | 6,304 MB |
| bistro-exterior | 1.2 | 15.2 | 0.6 | 2.8 | 19.8 s | 0.46 | 0.38 | 1.14 s | 4,476 MB |

- **Image decoding is the load.** tinygltf decodes every PNG through stb_image, one after another, while it reads the file: 27.6 of Intel Sponza's 31 seconds, 15.2 of Bistro's 19.8. The 72 4K PNGs of Intel Sponza's main file are 0.38 s each. The glTF parse itself (JSON and the binary buffers) is 1 to 2 s on these files, and building the vertex arrays and instances under a second.
- **MikkTSpace is 2.8 s on Bistro**, which ships normal maps without tangents; the other scenes either have tangents in the file or no normal map.
- **Init is 0.2 s at the floor and 1.1 s for the big scenes.** The floor is the CUDA context (the first runtime call, 87 ms in the Nsight trace) and the OptiX context (84 ms), plus 12 ms each for the acceleration structures and the pipeline. The big scenes add the texture upload (0.5 s for 4 to 6 GB of `cudaMemcpy2DToArray`) and 0.2 to 0.4 s of `optixAccelBuild` for 400 to 1,600 meshes, one GAS each.
- **GPU memory** is 516 MB for cornell at 1024x1024, of which the path and intersection buffers and their ping-pong spares are 230 MB (48 bytes per record, two records, two copies) and the rest the CUDA and OptiX contexts. Bistro and Intel Sponza hold 4 and 6 GB of textures on top.
- **Saving** is 0.25 s at 1024x1024: the download, the view transform on the host and the PNG encode.

Decoding only the images a material slot reads, across threads, is the fix the numbers point at; tinygltf can hand over the raw bytes and leave the decode to the loader.

#### Display path

The viewport under Nsight Systems, cornell at 1024x1024, 12 seconds: 988 frames, 10.9 ms each, against 8.63 ms per sample headless at the same settings (both with the sort on). Of the 2.3 ms a frame costs on top of its sample, 0.67 ms is `cudaGLMapBufferObject`, which maps the pixel buffer for CUDA every frame, and 56 us is `sendImageToPBO` (the view transform over a million pixels). The rest is the GL texture update and the swap.

#### Before and after

Two defaults changed from this profile, the sort and the error-check waits, both proven image-neutral with `tests/cmp_renders.sh` (52 of 52 renders byte-identical to the goldens from before this step). The same sweep against the new build, ms per sample:

| scene | 400x400 before | after | | 801x799 before | after | | 1024x1024 before | after | |
|---|---|---|---|---|---|---|---|---|---|
| cornell | 2.06 | 1.00 | 2.06x | 5.53 | 1.76 | 3.14x | 8.63 | 2.32 | 3.72x |
| cornell-box | 3.52 | 1.96 | 1.80x | 7.68 | 2.95 | 2.60x | 11.43 | 3.61 | 3.17x |
| cornell_helmet | 2.10 | 0.92 | 2.28x | 5.41 | 1.77 | 3.06x | 8.42 | 2.39 | 3.52x |
| transmission_roughness | 2.55 | 1.50 | 1.70x | 4.99 | 2.08 | 2.40x | 7.37 | 2.57 | 2.87x |
| dragon_attenuation | 3.12 | 1.63 | 1.91x | 5.30 | 2.39 | 2.22x | 7.52 | 3.14 | 2.39x |
| sponza | 2.46 | 1.31 | 1.88x | 6.67 | 2.74 | 2.43x | 10.20 | 3.98 | 2.56x |
| sponza_intel | 3.09 | 1.79 | 1.73x | 8.36 | 4.13 | 2.02x | 12.68 | 6.13 | 2.07x |
| bistro-exterior | 3.74 | 2.12 | 1.76x | 9.28 | 4.50 | 2.06x | 13.81 | 6.61 | 2.09x |

| Before | After, same axis |
|:---:|:---:|
| ![Before](img/readme/perf_stage_breakdown_1024x1024.png) | ![After](img/readme/perf_stage_breakdown_after_1024x1024.png) |

![Before and after](img/readme/perf_before_after_1024x1024.png)

Where a sample goes now, at 1024x1024: cornell is 2.32 ms, of which intersection 0.79, shading 0.69, compaction 0.62 and generation 0.20; Bistro is 6.61 ms with intersection 4.52, shading 1.23, compaction 0.62, generation 0.20. The stages now follow the scene, and the next things the numbers point at are, in order of what they would save:

1. Any-hit off for `BLEND` materials that are opaque in practice: up to 1.6 ms on Bistro, 0.8 on Intel Sponza.
2. Compaction at 0.6 ms: a CUB select over 48-byte records, at 45% of DRAM peak. Selecting indices and gathering, or compacting in place, would halve its traffic.
3. generateRayFromCamera at 0.2 ms: 8x8 blocks writing a struct field by field, at 27% of DRAM peak.
4. The alive-count readback, the last wait per bounce: about 0.1 ms per bounce at small path counts, which is the K-bounce raygen loop's whole budget.
5. Load: decoding only the images a material reads, in parallel, against 15 to 28 s of serial PNG decoding.

#### Compaction and sort: thrust to CUB with device LTO

Measured 2026-09-25, before the glTF materials, and kept here as the history of the stages above.

**The first version used thrust.** Compaction was `thrust::stable_partition`, the sort was `thrust::sort_by_key` over the path and intersection arrays together, and a `finalGather` kernel added every path's color to the image after the bounce loop. It produced the right image and spent most of the frame outside the kernels; the "Issues along the way" entry below has the numbers.

**The current version uses CUB with a workspace allocated once.** All scratch memory (spare path and intersection buffers, the sort's index arrays, CUB's temporary storage) is allocated in `wavefrontInit` at the largest path count the loop can see, so nothing is allocated during a sample.

* Compaction is a CUB select of the live paths into the spare buffer, then a pointer swap. Terminated paths are dropped rather than moved to the back, which is why light goes into the image inside `shadeMaterial` (today at the hit that emits it, see [Emission](#emission)) and `finalGather` is gone.
* The sort no longer moves the path structs through every radix pass. It sorts (material key, path index) pairs over only the bits the material count needs (one pass for Cornell instead of four), then gathers paths and intersections into the spare buffers once.

Both operations are out of place and ping-pong: the caller's `dev_paths` and `dev_intersections` point at a different allocation after every call, and the old one becomes the spare.

| Before and after, one sample | Buffer ownership through one bounce |
|:---:|:---:|
| ![Iteration before and after](img/readme/pathtrace_iteration_before_after.png) | ![Ping pong buffers](img/readme/path_buffer_ping_pong_one_bounce.png) |

**Results.** Cornell, headless, `--tonemap none`, median of 5 interleaved runs per cell, GPU otherwise idle. Every build produces a byte-identical PNG at the same settings (checked with `cmp`), so the rows differ in time only.

| build | 400x400, 100 spp | 800x800, 50 spp | 1024x1024, 50 spp |
|---|---|---|---|
| thrust | 4.96 | 14.86 | 22.90 |
| thrust + LTO | 3.69 | 9.54 | 14.08 |
| CUB + LTO (current) | 0.64 | 1.38 | 2.07 |

Each cell is ms per sample.

![ms per sample by build](img/readme/perf_ms_per_spp_1024x1024.png)

At 1024x1024 the frame went from 22.90 to 2.07 ms per sample, 11x. Device LTO is 1.6x of that on its own. The rest is the CUB rewrite: no allocation per call, no copy back, dead paths dropped instead of partitioned to the back.

### Issues along the way

Things that went wrong and what they turned out to be. Kept as a running log.

#### The OptiX cube kept the outward normal on exit hits

`boxIntersectionTest` returns the normal facing the ray on both entry and exit (on the exit face it stores the slab's entering normal, which points back inside). The OptiX cube hit program returned the face's outward normal in both cases. Nothing in Cornell starts a ray inside a wall, and the glass tests used the sphere, so the two paths agreed on every scene until the Box model test: a glass cube as a mesh rendered right on both paths, while the same cube as a scene cube came out dark under OptiX (mean 72 vs 78 of 255). The mesh hit program had been written with the flip, the cube one had not. One sign flip on the inside branch, and the cube scene matches the naive path to the same 4 levels as the mesh scenes.

600x600, 1000 spp, DEPTH 8, AgX, OptiX:

| Outward normal on exit | Normal facing the ray |
|:---:|:---:|
| ![Dark glass cube](img/bloopers/cornell_boxcube_optix_dark_glass_cube.png) | ![Fixed glass cube](img/readme/cornell_boxcube_optix_fixed.png) |

#### Schlick used the wrong cosine on exit

Schlick's approximation needs the cosine on the air side of the interface. Entering glass that is the incident angle. Leaving glass it is the transmitted angle, which the original code did not use. Real Fresnel reflectance hits 100% at the critical angle (about 42° for IOR 1.5), but with the incident cosine the curve did not get there until 90°, so exit rays just below critical leaked out instead of reflecting back inside. The fix computes `k = 1 - eta²(1 - cos²θ)` directly, uses `sqrt(k)` as the Schlick cosine when `eta > 1`, and treats `k < 0` as total internal reflection.

1000x1000, 500 spp, DEPTH 8, cropped to the sphere:

| Before | After |
|:---:|:---:|
| ![Schlick before](img/readme/schlick_before.png) | ![Schlick after](img/readme/schlick_after.png) |

Pixel difference, amplified 8x. The change is confined to the rim and the caustic under the sphere:

![Schlick diff](img/readme/schlick_diff_x8.png)

Notice that the fixed image is slightly *darker* at the rim. This is due to the fixed version keeps that light inside for more bounces, and with DEPTH 8 some of those longer paths hit the cap and get thrown away, in general this makes rays less likely to hit a light at the same depth. By changing the depth cap:

| Region | DEPTH 8 | DEPTH 32 |
|---|---|---|
| Sphere interior | 52.4 | 59.6 |
| Sphere rim | 40.4 | 44.7 |
| Floor caustic | 170.6 | 173.0 |
| Whole image | 25.3 | 27.8 |

We can see the depth cap was eating about 14% of the sphere's brightness on its own. The Schlick fix moved under 1%. This means glass materials needs a a lot higher depth than diffuse scenes. Honestly a surprise rabbit hole digging that landed this conclusion.

#### The sort and compaction were spending the frame in cudaMalloc

With the thrust version in, a 400x400 sample took 13 ms with the sort on and 5 ms with it off, and I assumed that was the sort itself. Timing each stage with CUDA events (Cornell 400x400, one sample, sort on, on the device LTO build so the kernels are at their real cost) said otherwise:

| stage | generate | intersect | sort | shade | compact | gather |
|---|---|---|---|---|---|---|
| ms | 0.08 | 0.36 | 8.9 | 0.21 | 4.0 | 0.03 |

The kernels add up to under 0.7 ms. The other 13 ms were in two thrust calls, for two reasons.

- Every thrust algorithm under the `thrust::device` policy allocates its scratch memory with `cudaMalloc` and frees it with `cudaFree` on the way out. `cudaFree` waits for all outstanding GPU work and `cudaMalloc` takes a device-wide lock. `stable_partition` is a composite (copy to a temporary, select the live paths to the front, write the dead ones to the back through a reverse iterator), each step with its own scratch, times eight bounces.
- `sort_by_key` over the zip iterator materializes the values into a temporary array of 68-byte tuples, radix-sorts 32-bit keys in four passes with every pass reading and writing every key and value, then copies back through the zip. Around 700 bytes of traffic per path per bounce, for a path that occupies 44 bytes.

Handing the same thrust calls a caching allocator (`thrust::cuda::par(alloc)`) removed 60 to 70% of both stages on its own, so that share was allocation. The rest was the copy-back and the extra passes, which is what the CUB version with a preallocated workspace and an index sort removes. Details and numbers in the performance section above.

#### Separable compilation left the intersection tests as real function calls

The base code's `CMakeLists.txt` sets `CUDA_SEPARABLE_COMPILATION ON`, which it needs so that `intersections.cu` and `interactions.cu` can be their own files. Without device link-time optimization, a `__device__` function defined in another file cannot be inlined into the kernel that calls it. `cuobjdump --dump-sass` on the exe showed `boxIntersectionTest`, `sphereIntersectionTest` and `scatterRay` as separate functions with `CALL.ABS` sites inside the kernels, and the call ABI shows up as local memory traffic:

| kernel | device calls | local memory instructions (STL/LDL) |
|---|---|---|
| `computeIntersections`, before | 3 | 72 |
| `shadeMaterial`, before | 4 | 38 |
| `computeIntersections`, with `-dlto` | 0 | 0 |
| `shadeMaterial`, with `-dlto` | 0 | 10 |

`Geom` is passed by value (three mat4), the output references need addresses, and the register copy of the path in `shadeMaterial` has to live in local memory because `scatterRay(PathSegment&)` takes its address across a call. So the "copy to registers, write back once" pattern in `shadeMaterial` was not happening at all.

![Device LTO before and after](img/readme/device_lto_inlining_before_after.png)

The fix is two lines in `CMakeLists.txt`: `-dlto` on the CUDA compile options and `$<DEVICE_LINK:-dlto>` on the device link options, Release only. Byte-identical image, and the no-sort frame got 25 to 40% faster depending on resolution, before any of the compaction work. The cost is at build time: the device link step takes about 4 s on every incremental build, since the inlining happens there.

#### The orbit camera mirrored any EYE off the z axis

The base code turns EYE into orbit angles with `acos`, which returns `[0, pi]` and drops the sign of x and of the pitch. So an eye at x = -4 was placed at x = +4 and an eye at height 8 ended up near the floor looking up. The rebuilt `right` and `up` were also not normalized (length cos(pitch)), which zoomed any pitched view in by 1/cos(pitch). The fix computes the angles from the eye's offset with `atan2`, normalizes the basis, and orders the loader so `view` exists before `right` is derived from it. Cornell's shipped camera sits on the z axis at lookAt height, the one spot where the old math was right, so its renders did not change.

400x400, 200 spp, AgX:

| EYE (-4, 5, 10.5) before | after | EYE (3, 8, 9) before | after |
|:---:|:---:|:---:|:---:|
| ![left before](img/readme/camera_left_before.png) | ![left after](img/readme/camera_left_after.png) | ![high before](img/readme/camera_high_before.png) | ![high after](img/readme/camera_high_after.png) |

In the same pass, `FOVY` was being read as a half angle (`tan(fovy)` where `tan(fovy/2)` belongs), so 45 in the scene file was a 90 degree vertical field. The loader now takes the full angle and the scene files say 90, which renders the same image and stops the value from capping when matching a real lens or a glTF camera.

#### Intel Sponza's dirt decals rendered as solid walls

Intel Sponza's courtyard walls carry `dirt_decal`, the file's one `alphaMode: BLEND` material, on three decal meshes (one per floor) laid over the walls. Its texture is dark grime (mean 75 of 255) with alpha under 0.5 on 70% of its texels, and the material multiplies alpha by 0.35 on top, so the decals should cover about 12% of the wall on average. The loader treated BLEND as opaque, so the full dark texture covered the walls and the stone only showed in the gaps between decals, which looked like dark brown plaster with patches broken off.

The fix is just actually support alpha blending: a hit counts with probability alpha, and otherwise the ray goes on, in the same any-hit program and naive-kernel test that handle [alpha mask](#alpha-mask). The random number is a hash of the pixel, iteration, depth, instance and triangle, so both intersection paths make the same decision for the same hit, and OptiX gets the same answer when it calls any-hit more than once for a triangle.

1280x720, 256 spp, DEPTH 8, AgX, OptiX:

| BLEND as opaque | BLEND as coverage |
|:---:|:---:|
| ![Decals as solid walls](img/bloopers/sponza_intel_blend_decals_opaque.png) | ![Decals as coverage](img/readme/sponza_intel_blend_decals_fixed.png) |

#### Smaller base code fixes

- **Every camera move freed and reallocated all device buffers.** A drag re-ran the full init per mouse event, about 2.8 ms at 800x800, more than a frame now costs. Buffers are allocated once; a camera change clears the image and resets the sample count.
- **The accumulation buffer was copied to the host every iteration** so the S key could save at any time. The copy now happens inside save.
- **A CUDA error waited on `getchar()` before exiting** on Windows, which hangs a headless run. Removed; the message goes to stderr either way.
