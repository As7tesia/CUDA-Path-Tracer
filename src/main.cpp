#include "glslUtility.hpp"
#include "image.h"
#include "pathtrace.h"
#include "scene.h"
#include "sceneStructs.h"
#include "utilities.h"

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "ImGui/imgui.h"
#include "ImGui/imgui_impl_glfw.h"
#include "ImGui/imgui_impl_opengl3.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

static std::string startTimeString;

// CLI options
static bool headless = false;
static std::string outPath;   // empty = default img/auto_saved/<FILE>.<time>.<spp>samp.png
static ToneMapMode toneMapMode = TONEMAP_AGX;
static float exposure = 1.f;

// The viewport camera. Yaw turns about world +Y: 0 looks down -Z, positive
// turns right, toward +X. Pitch tilts up from the horizon and stops just
// short of straight up or down, where a basis built against world +Y would
// not exist. The orbit pivot sits on the view axis, pivotDistance ahead, and
// travels with the camera.
struct ViewportPose
{
    glm::vec3 position;
    float yaw;
    float pitch;
    float pivotDistance;
};
static ViewportPose viewportPose;
static ViewportPose scenePose;  // the scene file's camera, restored by F
static float flySpeed;          // world units per second
static bool camchanged = true;

static const float PITCH_LIMIT = 0.5f * PI - 0.001f;
// Mouse look and orbit, radians per pixel of mouse motion
static const float TURN_PER_PIXEL = 0.0025f;
// The default fly speed crosses the scene's bounding box diagonal in this time
static const float SECONDS_TO_CROSS_SCENE = 4.0f;
// RMB + wheel scales the fly speed by this per notch
static const float SPEED_STEP = 1.25f;
// A wheel notch moves as far as this much flying
static const float WHEEL_STEP_SECONDS = 0.25f;
// LMB drag: a pixel of vertical motion moves as far as this much flying
static const float GROUND_SECONDS_PER_PIXEL = 0.005f;
// Alt + RMB drag: the distance to the pivot scales by e^-x for x pixels
static const float DOLLY_PER_PIXEL = 0.005f;
// A stalled frame (a slow render, the window waiting for input) moves the
// flying camera at most this far ahead
static const float MAX_FRAME_SECONDS = 0.25f;

// Mouse buttons held in the viewport (a press over an ImGui panel is ImGui's)
static bool leftMousePressed = false;
static bool rightMousePressed = false;
static bool middleMousePressed = false;
static double lastX;
static double lastY;

Scene* scene;
GuiDataContainer* guiData;
RenderState* renderState;
int iteration;

int width;
int height;

GLuint positionLocation = 0;
GLuint texcoordsLocation = 1;
GLuint pbo;
GLuint displayImage;

GLFWwindow* window;
GuiDataContainer* imguiData = NULL;
ImGuiIO* io = nullptr;

// Forward declarations for window loop and interactivity
void runCuda();
void runHeadless();
ViewportPose poseFromCamera(const Camera& cam);
void applyPose();
void fly(float seconds);
void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);
void mousePositionCallback(GLFWwindow* window, double xpos, double ypos);
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);

std::string currentTimeString()
{
    time_t now;
    time(&now);
    char buf[sizeof "0000-00-00_00-00-00z"];
    strftime(buf, sizeof buf, "%Y-%m-%d_%H-%M-%Sz", gmtime(&now));
    return std::string(buf);
}

//-------------------------------
//----------SETUP STUFF----------
//-------------------------------

void initTextures()
{
    glGenTextures(1, &displayImage);
    glBindTexture(GL_TEXTURE_2D, displayImage);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
}

void initVAO(void)
{
    GLfloat vertices[] = {
        -1.0f, -1.0f,
        1.0f, -1.0f,
        1.0f,  1.0f,
        -1.0f,  1.0f,
    };

    // The texture's first row is the image's top (v = 0 at the top of the
    // window) and its first column the image's left.
    GLfloat texcoords[] = {
        0.0f, 1.0f,
        1.0f, 1.0f,
        1.0f, 0.0f,
        0.0f, 0.0f
    };

    GLushort indices[] = { 0, 1, 3, 3, 1, 2 };

    GLuint vertexBufferObjID[3];
    glGenBuffers(3, vertexBufferObjID);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)positionLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(positionLocation);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(texcoords), texcoords, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)texcoordsLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(texcoordsLocation);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertexBufferObjID[2]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
}

GLuint initShader()
{
    const char* attribLocations[] = { "Position", "Texcoords" };
    GLuint program = glslUtility::createDefaultProgram(attribLocations, 2);
    GLint location;

    //glUseProgram(program);
    if ((location = glGetUniformLocation(program, "u_image")) != -1)
    {
        glUniform1i(location, 0);
    }

    return program;
}

void deletePBO(GLuint* pbo)
{
    if (pbo)
    {
        // unregister this buffer object with CUDA
        cudaGLUnregisterBufferObject(*pbo);

        glBindBuffer(GL_ARRAY_BUFFER, *pbo);
        glDeleteBuffers(1, pbo);

        *pbo = (GLuint)NULL;
    }
}

void deleteTexture(GLuint* tex)
{
    glDeleteTextures(1, tex);
    *tex = (GLuint)NULL;
}

void cleanupCuda()
{
    if (pbo)
    {
        deletePBO(&pbo);
    }
    if (displayImage)
    {
        deleteTexture(&displayImage);
    }
}

void initCuda()
{
    cudaGLSetGLDevice(0);
}

void initPBO()
{
    // set up vertex data parameter
    int num_texels = width * height;
    int num_values = num_texels * 4;
    int size_tex_data = sizeof(GLubyte) * num_values;

    // Generate a buffer ID called a PBO (Pixel Buffer Object)
    glGenBuffers(1, &pbo);

    // Make this the current UNPACK buffer (OpenGL is state-based)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

    // Allocate data for the buffer. 4-channel 8-bit image
    glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
    cudaGLRegisterBufferObject(pbo);
}

void errorCallback(int error, const char* description)
{
    fprintf(stderr, "%s\n", description);
}

bool init()
{
    glfwSetErrorCallback(errorCallback);

    if (!glfwInit())
    {
        exit(EXIT_FAILURE);
    }

    window = glfwCreateWindow(width, height, "CIS 565 Path Tracer", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetCursorPosCallback(window, mousePositionCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetScrollCallback(window, scrollCallback);
    // Unaccelerated mouse motion while a drag has the cursor disabled
    if (glfwRawMouseMotionSupported())
    {
        glfwSetInputMode(window, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
    }

    // Set up GL context
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK)
    {
        return false;
    }
    printf("Opengl Version:%s\n", glGetString(GL_VERSION));
    //Set up ImGui

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    io = &ImGui::GetIO(); (void)io;
    ImGui::StyleColorsLight();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 120");

    // Initialize other stuff
    initVAO();
    initTextures();
    initCuda();
    initPBO();
    GLuint passthroughProgram = initShader();

    glUseProgram(passthroughProgram);
    glActiveTexture(GL_TEXTURE0);

    return true;
}

void InitImguiData(GuiDataContainer* guiData)
{
    imguiData = guiData;
}


// LOOK: Un-Comment to check ImGui Usage
void RenderImGui()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    bool show_demo_window = true;
    bool show_another_window = false;
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
    static float f = 0.0f;
    static int counter = 0;

    ImGui::Begin("Path Tracer Analytics");                  // Create a window called "Hello, world!" and append into it.
    
    // LOOK: Un-Comment to check the output window and usage
    //ImGui::Text("This is some useful text.");               // Display some text (you can use a format strings too)
    //ImGui::Checkbox("Demo Window", &show_demo_window);      // Edit bools storing our window open/close state
    //ImGui::Checkbox("Another Window", &show_another_window);

    //ImGui::SliderFloat("float", &f, 0.0f, 1.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
    //ImGui::ColorEdit3("clear color", (float*)&clear_color); // Edit 3 floats representing a color

    //if (ImGui::Button("Button"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
    //    counter++;
    //ImGui::SameLine();
    //ImGui::Text("counter = %d", counter);
    ImGui::Text("Traced Depth %d", imguiData->TracedDepth);
    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    ImGui::Text("Fly speed %.3g units/s (RMB + wheel)", flySpeed);
    ImGui::End();


    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

}

void mainLoop()
{
    double lastFrameTime = glfwGetTime();
    while (!glfwWindowShouldClose(window))
    {
        // A finished render has nothing left to compute: wait for input
        // instead of redrawing the same image as fast as the GPU can.
        const bool finished = !camchanged && iteration >= (int)renderState->iterations;
        if (finished)
        {
            glfwWaitEventsTimeout(0.1);
        }
        else
        {
            glfwPollEvents();
        }

        const double now = glfwGetTime();
        fly((float)std::min(now - lastFrameTime, (double)MAX_FRAME_SECONDS));
        lastFrameTime = now;

        runCuda();

        std::string title = "CIS565 Path Tracer | " + utilityCore::convertIntToString(iteration) + " Iterations";
        glfwSetWindowTitle(window, title.c_str());
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBindTexture(GL_TEXTURE_2D, displayImage);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glClear(GL_COLOR_BUFFER_BIT);

        // Binding GL_PIXEL_UNPACK_BUFFER back to default
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        // VAO, shader program, and texture already bound
        glDrawElements(GL_TRIANGLES, 6,  GL_UNSIGNED_SHORT, 0);

        // Render ImGui Stuff
        RenderImGui();

        glfwSwapBuffers(window);
    }

    pathtraceFree();
    cleanupCuda();

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
    cudaDeviceReset();
}

//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv)
{
    startTimeString = currentTimeString();

    const char* usage =
        "Usage: %s SCENEFILE [--headless] [--spp N] [--res WxH] [--depth N] [--out PATH.png]\n"
        "                     [--no-rr] [--no-sort] [--tonemap none|aces|agx|agx-punchy] [--exposure X]\n"
        "                     [--no-optix] [--optix-validate]\n"
        "  SCENEFILE        a scene .json, or a .gltf / .glb file that is the whole scene\n"
        "  --headless       render without a window and exit after saving\n"
        "  --spp N          override the scene's ITERATIONS\n"
        "  --res WxH        override the scene's RES\n"
        "  --depth N        override the scene's DEPTH, the most rays a path may trace\n"
        "  --out PATH       write exactly this file (default: img/auto_saved/<FILE>.<time>.<spp>samp.png)\n"
        "  --no-rr          disable Russian roulette path termination\n"
        "  --no-sort        disable sorting paths by material before shading\n"
        "  --no-optix       intersect with the naive per-object kernel instead of OptiX\n"
        "  --optix-validate OptiX validation mode: checks every launch, slow\n"
        "  --tonemap MODE   view transform for display and PNG (default agx; none = raw clamp)\n"
        "  --exposure X     linear multiplier before the view transform (default 1.0)\n";

    const char* sceneFile = nullptr;
    SceneOverrides ov;
    for (int i = 1; i < argc; i++)
    {
        std::string a = argv[i];
        auto needValue = [&](const char* flag) -> const char* {
            if (i + 1 >= argc)
            {
                printf("%s needs a value\n", flag);
                exit(1);
            }
            return argv[++i];
        };
        if (a == "--headless")
        {
            headless = true;
        }
        else if (a == "--spp")
        {
            ov.iterations = atoi(needValue("--spp"));
        }
        else if (a == "--res")
        {
            if (sscanf(needValue("--res"), "%dx%d", &ov.width, &ov.height) != 2)
            {
                printf("--res expects WxH, e.g. 800x600\n");
                return 1;
            }
        }
        else if (a == "--depth")
        {
            ov.traceDepth = atoi(needValue("--depth"));
        }
        else if (a == "--out")
        {
            outPath = needValue("--out");
        }
        else if (a == "--no-rr")
        {
            setRussianRoulette(false);
        }
        else if (a == "--no-sort")
        {
            setMaterialSort(false);
        }
        else if (a == "--no-optix")
        {
            setOptix(false);
        }
        else if (a == "--optix-validate")
        {
            setOptixValidation(true);
        }
        else if (a == "--tonemap")
        {
            std::string m = needValue("--tonemap");
            if (m == "none")      toneMapMode = TONEMAP_NONE;
            else if (m == "aces") toneMapMode = TONEMAP_ACES;
            else if (m == "agx")  toneMapMode = TONEMAP_AGX;
            else if (m == "agx-punchy") toneMapMode = TONEMAP_AGX_PUNCHY;
            else
            {
                printf("--tonemap expects none, aces, agx or agx-punchy\n");
                return 1;
            }
        }
        else if (a == "--exposure")
        {
            exposure = (float)atof(needValue("--exposure"));
        }
        else if (a.size() > 2 && a.substr(0, 2) == "--")
        {
            printf("Unknown option %s\n", argv[i]);
            printf(usage, argv[0]);
            return 1;
        }
        else
        {
            sceneFile = argv[i];
        }
    }

    if (sceneFile == nullptr)
    {
        printf(usage, argv[0]);
        return 1;
    }

    setToneMap(toneMapMode, exposure);

    // Load scene file
    scene = new Scene(sceneFile, ov);

    //Create Instance for ImGUIData
    guiData = new GuiDataContainer();

    // Set up camera stuff from loaded path tracer settings
    iteration = 0;
    renderState = &scene->state;
    Camera& cam = renderState->camera;
    width = cam.resolution.x;
    height = cam.resolution.y;

    // The viewport starts at the scene file's camera. Headless renders take
    // their basis from the same pose, so they match the window's first frame.
    scenePose = poseFromCamera(cam);
    viewportPose = scenePose;
    applyPose();

    if (headless)
    {
        runHeadless();
        return 0;
    }

    glm::vec3 lo, hi;
    scene->bounds(0, scene->geoms.size(), lo, hi);
    flySpeed = (lo.x <= hi.x ? glm::length(hi - lo) : 1.0f) / SECONDS_TO_CROSS_SCENE;

    // Initialize CUDA and GL components
    init();
    // Device buffers live for the whole run; camera changes only clear the image
    pathtraceInit(scene);

    // Initialize ImGui Data
    InitImguiData(guiData);
    InitDataContainer(guiData);

    // GLFW main loop
    mainLoop();

    return 0;
}

void saveImage()
{
    float samples = iteration;
    pathtraceDownloadImage();
    // output image file
    Image img(width, height);

    for (int x = 0; x < width; x++)
    {
        for (int y = 0; y < height; y++)
        {
            int index = x + (y * width);
            glm::vec3 pix = renderState->image[index] / samples;   // scene-linear average
            img.setPixel(x, y, applyToneMap(pix, toneMapMode, exposure));
        }
    }

    std::string filename;
    if (!outPath.empty())
    {
        // savePNG appends ".png" itself
        filename = outPath;
        if (filename.size() > 4 && filename.substr(filename.size() - 4) == ".png")
        {
            filename = filename.substr(0, filename.size() - 4);
        }
    }
    else
    {
        filename = "img/auto_saved/" + renderState->imageName;
        std::ostringstream ss;
        ss << filename << "." << startTimeString << "." << samples << "samp";
        filename = ss.str();
    }

    std::filesystem::path dir = std::filesystem::path(filename).parent_path();
    if (!dir.empty())
    {
        std::filesystem::create_directories(dir);
    }

    // CHECKITOUT
    img.savePNG(filename);
    //img.saveHDR(filename);  // Save a Radiance HDR file
}

// Headless mode: no GLFW / GL / ImGui / PBO. Render state.iterations samples,
// save, and exit. Used for agent-driven test renders.
void runHeadless()
{
    pathtraceInit(scene);

    auto t0 = std::chrono::steady_clock::now();
    for (iteration = 1; iteration <= renderState->iterations; iteration++)
    {
        pathtrace(nullptr, 0, iteration);
    }
    cudaDeviceSynchronize();
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    iteration = renderState->iterations;
    saveImage();
    pathtraceFree();
    cudaDeviceReset();

    printf("headless: %dx%d, %d spp, %.1f ms total, %.2f ms/spp\n",
        width, height, renderState->iterations, ms, ms / renderState->iterations);
}

glm::vec3 viewDirection(const ViewportPose& p)
{
    return glm::vec3(std::sin(p.yaw) * std::cos(p.pitch), std::sin(p.pitch), -std::cos(p.yaw) * std::cos(p.pitch));
}

// The pose that sees what the loaded camera sees: the same eye and view
// direction, with lookAt as the pivot. A camera looking straight up or down
// (a glTF camera can) is tilted a hair off the pole.
ViewportPose poseFromCamera(const Camera& cam)
{
    ViewportPose p;
    p.position = cam.position;
    p.yaw = std::atan2(cam.view.x, -cam.view.z);
    p.pitch = glm::clamp(std::asin(glm::clamp(cam.view.y, -1.0f, 1.0f)), -PITCH_LIMIT, PITCH_LIMIT);
    p.pivotDistance = glm::length(cam.lookAt - cam.position);
    return p;
}

// Writes viewportPose into the render camera: the position, the basis
// generateRayFromCamera reads, and the pivot as lookAt. The basis is built
// against world +Y, so the scene's UP only matters to the loader's basis,
// which this replaces before the first pathtrace.
void applyPose()
{
    Camera& cam = renderState->camera;
    cam.position = viewportPose.position;
    cam.view = viewDirection(viewportPose);
    // Both normalized: cross(view, +Y) has length cos(pitch), and the pixel
    // offsets in generateRayFromCamera scale by right and up directly.
    const glm::vec3 right = glm::normalize(glm::cross(cam.view, glm::vec3(0.0f, 1.0f, 0.0f)));
    cam.up = glm::normalize(glm::cross(right, cam.view));
    cam.right = cam.mirrored ? -right : right;
    cam.lookAt = viewportPose.position + viewportPose.pivotDistance * cam.view;
}

void runCuda()
{
    if (camchanged)
    {
        iteration = 0;
        camchanged = false;
    }

    // Map OpenGL buffer object for writing from CUDA on a single GPU
    // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer

    if (iteration == 0)
    {
        pathtraceReset();
    }

    if (iteration < renderState->iterations)
    {
        uchar4* pbo_dptr = NULL;
        iteration++;
        cudaGLMapBufferObject((void**)&pbo_dptr, pbo);

        // execute the kernel
        int frame = 0;
        pathtrace(pbo_dptr, frame, iteration);

        // unmap buffer object
        cudaGLUnmapBufferObject(pbo);

        // The last sample: save the image. The window stays open on the
        // finished render until Escape; moving the camera starts it over.
        if (iteration == (int)renderState->iterations)
        {
            saveImage();
        }
    }
}

//-------------------------------
//------INTERACTIVITY SETUP------
//-------------------------------

// Every change to viewportPose ends here: the render camera follows it and
// the image starts over.
void cameraMoved()
{
    applyPose();
    camchanged = true;
}

glm::vec3 pivotOf(const ViewportPose& p)
{
    return p.position + p.pivotDistance * viewDirection(p);
}

bool altHeld()
{
    return glfwGetKey(window, GLFW_KEY_LEFT_ALT) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_RIGHT_ALT) == GLFW_PRESS;
}

// Mouse right turns toward the right of the screen, mouse up looks up. A
// mirrored camera's screen right is its world left, hence the sign.
void turn(double dx, double dy)
{
    const float screenRight = renderState->camera.mirrored ? -1.0f : 1.0f;
    viewportPose.yaw = std::remainder(viewportPose.yaw + screenRight * TURN_PER_PIXEL * (float)dx, 2.0f * PI);
    viewportPose.pitch = glm::clamp(viewportPose.pitch - TURN_PER_PIXEL * (float)dy, -PITCH_LIMIT, PITCH_LIMIT);
}

// Alt + LMB: turn, then move the camera back onto the line through the
// pivot, so the pivot stays where it is on screen.
void orbit(double dx, double dy)
{
    const glm::vec3 pivot = pivotOf(viewportPose);
    turn(dx, dy);
    viewportPose.position = pivot - viewportPose.pivotDistance * viewDirection(viewportPose);
}

// Alt + RMB: mouse right or up moves toward the pivot. The distance scales
// rather than shrinking by a fixed step, so the camera never reaches it.
void dollyToPivot(double dx, double dy)
{
    const glm::vec3 pivot = pivotOf(viewportPose);
    viewportPose.pivotDistance *= std::exp(-DOLLY_PER_PIXEL * (float)(dx - dy));
    viewportPose.position = pivot - viewportPose.pivotDistance * viewDirection(viewportPose);
}

// MMB, LMB + RMB, Alt + MMB: the camera moves with the mouse in its own
// plane, one pixel's width at the pivot's distance per pixel.
void pan(double dx, double dy)
{
    const Camera& cam = renderState->camera;
    const float perPixel = viewportPose.pivotDistance * cam.pixelLength.y;
    viewportPose.position += perPixel * ((float)dx * cam.right - (float)dy * cam.up);
}

// LMB: mouse up and down move forward and back along the level heading,
// mouse left and right turn.
void moveAlongGround(double dx, double dy)
{
    turn(dx, 0.0);
    const glm::vec3 heading(std::sin(viewportPose.yaw), 0.0f, -std::cos(viewportPose.yaw));
    viewportPose.position -= (float)dy * flySpeed * GROUND_SECONDS_PER_PIXEL * heading;
}

// RMB + W A S D E Q. The keys are read every frame rather than from key
// events, so a held key moves the camera smoothly. Not with Alt, where RMB
// dollies.
void fly(float seconds)
{
    if (!rightMousePressed || altHeld() || io->WantCaptureKeyboard)
    {
        return;
    }
    const Camera& cam = renderState->camera;
    auto held = [](int key) { return glfwGetKey(window, key) == GLFW_PRESS; };
    const glm::vec3 direction = (float)(held(GLFW_KEY_W) - held(GLFW_KEY_S)) * cam.view
        + (float)(held(GLFW_KEY_D) - held(GLFW_KEY_A)) * cam.right
        + (float)(held(GLFW_KEY_E) - held(GLFW_KEY_Q)) * glm::vec3(0.0f, 1.0f, 0.0f);
    if (direction == glm::vec3(0.0f))
    {
        return;
    }
    viewportPose.position += flySpeed * seconds * glm::normalize(direction);
    cameraMoved();
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (action != GLFW_PRESS || io->WantCaptureKeyboard)
    {
        return;
    }
    switch (key)
    {
        case GLFW_KEY_ESCAPE:
            saveImage();
            glfwSetWindowShouldClose(window, GL_TRUE);
            break;
        case GLFW_KEY_S:
            // Plain S flies backward
            if (mods & GLFW_MOD_CONTROL)
            {
                saveImage();
            }
            break;
        case GLFW_KEY_F:
            // Back to the scene file's camera. Not while flying, where F sits
            // next to the movement keys.
            if (!rightMousePressed)
            {
                viewportPose = scenePose;
                cameraMoved();
            }
            break;
    }
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    const bool press = action == GLFW_PRESS;
    // A press over an ImGui panel is ImGui's. Releases always count, so a
    // drag that ends over a panel does not leave its button held.
    if (press && io->WantCaptureMouse)
    {
        return;
    }
    switch (button)
    {
        case GLFW_MOUSE_BUTTON_LEFT:
            leftMousePressed = press;
            break;
        case GLFW_MOUSE_BUTTON_RIGHT:
            rightMousePressed = press;
            break;
        case GLFW_MOUSE_BUTTON_MIDDLE:
            middleMousePressed = press;
            break;
        default:
            return;
    }
    // A drag hides and locks the cursor, so it never stops at the edge of the
    // window or the screen. GLFW puts the cursor back where it was afterwards.
    const bool dragging = leftMousePressed || rightMousePressed || middleMousePressed;
    glfwSetInputMode(window, GLFW_CURSOR, dragging ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
    glfwGetCursorPos(window, &lastX, &lastY);
}

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos)
{
    const double dx = xpos - lastX;
    const double dy = ypos - lastY;
    lastX = xpos;
    lastY = ypos;
    if (dx == 0.0 && dy == 0.0)
    {
        return;
    }

    const bool alt = altHeld();
    if (alt && leftMousePressed)
    {
        orbit(dx, dy);
    }
    else if (alt && rightMousePressed)
    {
        dollyToPivot(dx, dy);
    }
    else if (middleMousePressed || (leftMousePressed && rightMousePressed))
    {
        pan(dx, dy);
    }
    else if (rightMousePressed)
    {
        turn(dx, dy);
    }
    else if (leftMousePressed)
    {
        moveAlongGround(dx, dy);
    }
    else
    {
        return;
    }
    cameraMoved();
}

// The wheel moves the camera along the view in steps. With RMB held it sets
// the fly speed instead.
void scrollCallback(GLFWwindow* window, double xoffset, double yoffset)
{
    if (io->WantCaptureMouse)
    {
        return;
    }
    if (rightMousePressed)
    {
        flySpeed *= std::pow(SPEED_STEP, (float)yoffset);
        return;
    }
    viewportPose.position += (float)yoffset * flySpeed * WHEEL_STEP_SECONDS * viewDirection(viewportPose);
    cameraMoved();
}
