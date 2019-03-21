cmake_minimum_required(VERSION 3.9)
if (APPLE)
    set(CMAKE_C_COMPILER /usr/local/Cellar/llvm/5.0.0/bin/clang)
    set(CMAKE_CXX_COMPILER /usr/local/Cellar/llvm/5.0.0/bin/clang++)
elseif (MSVC)
    set(CMAKE_TOOLCHAIN_FILE:PATH C:Utilities/vcpkg/scripts/buildsystems/vcpkg.cmake)
endif ()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_VERBOSE_MAKEFILE ON)

project(PBF_15418)
option(USE_CUDA "enable CUDA in compilation" ON)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config.h.in ${CMAKE_CURRENT_BINARY_DIR}/config.h)

if (USE_CUDA)
    set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/bin/nvcc.exe")
    enable_language(CUDA)
    message("CUDA enabled")

    if (NOT DEFINED CMAKE_CUDA_STANDARD)
        set(CMAKE_CUDA_STANDARD 14)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    endif ()
endif ()

find_package(GLFW3 REQUIRED)
find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glm REQUIRED)

if (APPLE)
    find_library(COCOA_LIBRARY Cocoa REQUIRED)
endif ()

add_library(particle_system_frontend
        glWindow.cpp
        glRenderer.cpp
        ParticleSystemSerial.cpp)

target_include_directories(particle_system_frontend
        PUBLIC
        ${GLEW_INCLUDE_DIRS}
        ${OPENGL_INCLUDE_DIR}
        ${CMAKE_CURRENT_BINARY_DIR})

if (APPLE)
    target_link_libraries(particle_system_frontend
            ${COCOA_LIBRARY}
            ${CMAKE_SOURCE_DIR}/lib/macos/libsoil2.a)
elseif (MSVC)
    if (${CMAKE_BUILD_TYPE})

    endif ()
    target_link_libraries(particle_system_frontend
            ${CMAKE_SOURCE_DIR}/lib/windows/soil2-debug.lib)
else ()
    target_link_libraries(particle_system_frontend
            ${CMAKE_SOURCE_DIR}/lib/linux/libsoil2.a)
endif ()

target_link_libraries(particle_system_frontend
        glfw
        ${GLEW_LIBRARIES}
        ${OPENGL_LIBRARIES})

if (MSVC)
    target_compile_options(particle_system_frontend PRIVATE
            /openmp
            )
else ()
    target_compile_options(particle_system_frontend PRIVATE
            -fopenmp
            )
endif ()

if (USE_CUDA)
    set(executable_files
            ParticleSystemCUDA.cu
            ParticleSystemCUDAUtils.cu
            main.cpp)
else ()
    set(executable_files main.cpp)
endif ()

add_executable(PBF_15418 ${executable_files})
target_include_directories(PBF_15418 PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_dependencies(PBF_15418 particle_system_frontend)
target_link_libraries(PBF_15418 particle_system_frontend)
