cmake_minimum_required(VERSION 3.9)

if (WIN32)
    set(CMAKE_TOOLCHAIN_FILE "C:/Utilities/vcpkg/scripts/buildsystems/vcpkg.cmake")
	message("On Windows, vcpkg toolchain file: " ${CMAKE_TOOLCHAIN_FILE})
endif ()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_VERBOSE_MAKEFILE ON)

project(PBF_15418)
option(USE_CUDA "enable CUDA in compilation" ON)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config.h.in ${CMAKE_CURRENT_BINARY_DIR}/config.h)

if (USE_CUDA)
    enable_language(CUDA)
    message("CUDA enabled")

    if (NOT DEFINED CMAKE_CUDA_STANDARD)
        set(CMAKE_CUDA_STANDARD 14)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    endif ()
endif ()

find_package(glfw3 REQUIRED)
find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glm REQUIRED)

if (APPLE)
    find_library(COCOA_LIBRARY Cocoa REQUIRED)
endif ()

add_library(frontend glWindow.cpp glRenderer.cpp)

target_include_directories(frontend
        PUBLIC
        ${GLEW_INCLUDE_DIRS}
        ${OPENGL_INCLUDE_DIR})

target_link_libraries(frontend
        glfw
        glm
        ${GLEW_LIBRARIES}
        ${OPENGL_LIBRARIES})

add_library(particle_system_cpu ParticleSystemSerial.cpp)

if (MSVC)
    target_compile_options(particle_system_cpu PRIVATE /openmp)
else ()
    target_compile_options(particle_system_cpu PRIVATE -fopenmp)
endif ()

add_executable(PBF_15418 main.cpp)
target_include_directories(PBF_15418 PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

target_link_libraries(PBF_15418 particle_system_cpu frontend)

if (USE_CUDA)
    add_library(particle_system_gpu ParticleSystemCUDA.cu ParticleSystemCUDAUtils.cu)

    target_link_libraries(PBF_15418 particle_system_gpu)
endif ()
