cmake_minimum_required(VERSION 3.24)
project(cis565_path_tracer LANGUAGES CUDA CXX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Enable C++17 for host code
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# CUDA 13's CCCL headers (Thrust/CUB) refuse to build with MSVC's traditional
# preprocessor; they require the standards-conforming one.
if(MSVC)
    add_compile_options(
        "$<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>"
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>")
endif()

find_package(OpenGL REQUIRED)

if(UNIX)
    include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
    find_package(glfw3 REQUIRED)
    find_package(GLEW REQUIRED)
    set(GL_LIBRARIES glfw ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES})
else(UNIX)
    set(EXTERNAL "${CMAKE_SOURCE_DIR}/external")

    set(GLFW_ROOT_DIR ${EXTERNAL})
    set(GLFW_USE_STATIC_LIBS ON)
    find_package(GLFW REQUIRED)

    set(GLEW_ROOT_DIR ${EXTERNAL})
    set(GLEW_USE_STATIC_LIBS ON)
    find_package(GLEW REQUIRED)

    add_definitions(${GLEW_DEFINITIONS})
    include_directories(${GLEW_INCLUDE_DIR} ${GLFW_INCLUDE_DIR})
    set(GL_LIBRARIES ${GLEW_LIBRARY} ${GLFW_LIBRARY} ${OPENGL_LIBRARY})
endif(UNIX)

set(GLM_ROOT_DIR "${CMAKE_SOURCE_DIR}/external")
find_package(GLM REQUIRED)
include_directories(${GLM_INCLUDE_DIRS})

# OptiX. The API headers are vendored under external/include/optix (from the
# OptiX SDK 9.1.0). The implementation lives in the display driver and is
# loaded at run time by optixInit(), so there is nothing to link. Host code is
# src/optix_intersect.cpp; the device programs are src/optix_programs.cu, built
# separately below.
set(OPTIX_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/external/include/optix")
include_directories(${OPTIX_INCLUDE_DIR})

set(headers
    src/gltf_loader.h
    src/image.h
    src/interactions.h
    src/intersections.h
    src/glslUtility.hpp
    src/optix_intersect.h
    src/optix_params.h
    src/pathtrace.h
    src/scene.h
    src/sceneStructs.h
    src/textures.h
    src/tonemap.h
    src/utilities.h
    src/wavefront_ops.h
)

set(sources
    src/main.cpp
    src/stb.cpp
    src/gltf_loader.cpp
    src/image.cpp
    src/glslUtility.cpp
    src/pathtrace.cu
    src/intersections.cu
    src/interactions.cu
    src/optix_intersect.cpp
    src/scene.cpp
    src/textures.cpp
    src/utilities.cpp
    src/wavefront_ops.cu
)

set(imgui_headers
    src/ImGui/imconfig.h
    src/ImGui/imgui.h
    src/ImGui/imgui_impl_glfw.h
    src/ImGui/imgui_impl_opengl3.h
    src/ImGui/imgui_impl_opengl3_loader.h
    src/ImGui/imgui_internal.h
    src/ImGui/imstb_rectpack.h
    src/ImGui/imstb_textedit.h
    src/ImGui/imstb_truetype.h
)

set(imgui_sources
    src/ImGui/imgui.cpp
    src/ImGui/imgui_demo.cpp
    src/ImGui/imgui_draw.cpp
    src/ImGui/imgui_impl_glfw.cpp
    src/ImGui/imgui_impl_opengl3.cpp
    src/ImGui/imgui_tables.cpp
    src/ImGui/imgui_widgets.cpp
)

list(SORT headers)
list(SORT sources)
list(SORT imgui_headers)
list(SORT imgui_sources)

source_group("Headers" FILES ${headers})
source_group("Sources" FILES ${sources})
source_group("ImGui\\Headers" FILES ${imgui_headers})
source_group("ImGui\\Sources" FILES ${imgui_sources})

#add_subdirectory(src/ImGui)
#add_subdirectory(stream_compaction)  # TODO: uncomment if using your stream compaction

add_executable(${CMAKE_PROJECT_NAME} ${sources} ${headers} ${imgui_sources} ${imgui_headers})
target_link_libraries(${CMAKE_PROJECT_NAME}
    ${GL_LIBRARIES}
    #stream_compaction  # TODO: uncomment if using your stream compaction
)
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES native)

target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE "$<$<AND:$<CONFIG:Debug,RelWithDebInfo>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")
target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo;-src-in-ptx>")

# Device link-time optimization. Separable compilation compiles each .cu on its
# own, so device functions defined in another file (the intersection tests,
# scatterRay) stay real calls inside the kernels unless the device link step
# can inline them. Release only; Debug and RelWithDebInfo use -G.
target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-dlto>")
target_link_options(${CMAKE_PROJECT_NAME} PRIVATE "$<$<CONFIG:Release>:$<DEVICE_LINK:-dlto>>")

# OptiX device programs. They are not linked like the other .cu files: the
# driver compiles them at run time from OptiX-IR. nvcc emits that IR here,
# cmake/EmbedFile.cmake turns the file into a byte array, and the array is
# compiled into the executable, so optixModuleCreate reads the programs from
# memory rather than from a file next to the binary. sm_75 is the oldest RTX
# generation; the driver specializes the IR for the GPU it runs on.
set(optix_programs "${CMAKE_SOURCE_DIR}/src/optix_programs.cu")
set(optix_ir "${CMAKE_BINARY_DIR}/optix_programs.optixir")
set(optix_ir_cpp "${CMAKE_BINARY_DIR}/optix_programs_ir.cpp")
add_custom_command(
    OUTPUT ${optix_ir}
    COMMAND ${CMAKE_CUDA_COMPILER} -optix-ir -arch=sm_75 -rdc=true -std=c++17 -m64 -lineinfo
            -ccbin "${CMAKE_CXX_COMPILER}"
            "-I${OPTIX_INCLUDE_DIR}" "-I${GLM_INCLUDE_DIRS}" "-I${CMAKE_SOURCE_DIR}/src"
            -o ${optix_ir} ${optix_programs}
    DEPENDS ${optix_programs}
            "${CMAKE_SOURCE_DIR}/src/optix_params.h"
            "${CMAKE_SOURCE_DIR}/src/sceneStructs.h"
    COMMENT "nvcc -optix-ir optix_programs.cu"
    VERBATIM)
add_custom_command(
    OUTPUT ${optix_ir_cpp}
    COMMAND ${CMAKE_COMMAND} -DINPUT=${optix_ir} -DOUTPUT=${optix_ir_cpp} -DSYMBOL=optix_programs_ir
            -P "${CMAKE_SOURCE_DIR}/cmake/EmbedFile.cmake"
    DEPENDS ${optix_ir} "${CMAKE_SOURCE_DIR}/cmake/EmbedFile.cmake"
    COMMENT "embedding optix_programs.optixir"
    VERBATIM)
target_sources(${CMAKE_PROJECT_NAME} PRIVATE ${optix_ir_cpp} ${optix_programs})
# Listed so it shows in the IDE; nvcc must not build it as an ordinary CUDA file.
set_source_files_properties(${optix_programs} PROPERTIES HEADER_FILE_ONLY ON)
source_group("OptiX" FILES ${optix_programs})

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${CMAKE_PROJECT_NAME})

