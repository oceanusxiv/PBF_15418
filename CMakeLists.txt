cmake_minimum_required(VERSION 3.2)
# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_C_COMPILER /usr/local/Cellar/llvm/5.0.0/bin/clang)
set(CMAKE_CXX_COMPILER /usr/local/Cellar/llvm/5.0.0/bin/clang++)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
project(PBF_15418)

#find_package(glfw3 3.2 REQUIRED)
find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)
find_package(PkgConfig)
find_package(Gperftools)
if (!APPLE)
find_package(CUDA REQUIRED)
endif (!APPLE)

pkg_search_module(GLFW REQUIRED glfw3)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
SET(GCC_COVERAGE_COMPILE_FLAGS "-fopenmp -Ofast")
SET(GCC_COVERAGE_LINK_FLAGS    "-Wl,-no_pie")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${GCC_COVERAGE_LINK_FLAGS}")

if(APPLE)
    INCLUDE_DIRECTORIES ( /System/Library/Frameworks )
    FIND_LIBRARY(COCOA_LIBRARY Cocoa)
    MARK_AS_ADVANCED (COCOA_LIBRARY)
    SET(EXTRA_LIBS ${COCOA_LIBRARY})
else(APPLE)
    include_directories(/afs/andrew.cmu.edu/usr8/efang/public/glm/)
endif (APPLE)

include_directories(${GLFW_INCLUDE_DIRS})
include_directories(${GLEW_INCLUDE_DIRS})
include_directories(${OPENGL_INCLUDE_DIR})
include_directories(${Gperftools_INCLUDE_DIRS})
if (!APPLE)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_52,code=sm_52 -gencode arch=compute_61,code=sm_61 -std=c++11)
endif (!APPLE)
#set(CUDA_LIBRARY saxpy)
#set(CUDA_SOURCES saxpy.cu)
#cuda_add_library(${CUDA_LIBRARY} ${CUDA_SOURCES})

if (APPLE)
    set(SOURCE_FILES main.cpp glWindow.cpp glWindow.h shader.h glRenderer.cpp glRenderer.h ParticleSystemSerial.cpp
        ParticleSystemSerial.h ParticleSystem.h Camera.h SOIL2.h Particle.h)
    add_executable(PBF_15418 ${SOURCE_FILES})
else (APPLE)
    set(SOURCE_FILES main.cpp glWindow.cpp glWindow.h shader.h glRenderer.cpp glRenderer.h ParticleSystemSerial.cpp
        ParticleSystemSerial.h ParticleSystem.h Camera.h SOIL2.h Particle.h ParticleSystemCUDA.h ParticleSystemCUDA.cpp)
    cuda_add_executable(PBF_15418 ${SOURCE_FILES} ParticleSystemCUDA.cu)
endif (APPLE)

if(APPLE)
    target_link_libraries(PBF_15418 ${EXTRA_LIBS})
    set(SOIL2_DIR ${CMAKE_SOURCE_DIR}/lib/libsoil2.a)
else(APPLE)
    set(SOIL2_DIR ${CMAKE_SOURCE_DIR}/lib/linux/libsoil2.a)
endif(APPLE)

target_link_libraries(PBF_15418 
${CUDA_LIBRARIES} 
${CUDA_LIBRARY} 
${GLFW_LIBRARIES} 
${GLEW_LIBRARIES} 
${OPENGL_LIBRARIES} 
${SOIL2_DIR}
profiler)
