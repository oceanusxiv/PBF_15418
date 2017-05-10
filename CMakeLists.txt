cmake_minimum_required(VERSION 2.8)
project(PBF_15418)

#find_package(glfw3 3.2 REQUIRED)
find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)
find_package( PkgConfig )
find_package(CUDA REQUIRED)

pkg_search_module(GLFW REQUIRED glfw3)

set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

if(APPLE)
    INCLUDE_DIRECTORIES ( /System/Library/Frameworks )
    FIND_LIBRARY(COCOA_LIBRARY Cocoa)
    MARK_AS_ADVANCED (COCOA_LIBRARY)
    SET(EXTRA_LIBS ${COCOA_LIBRARY})
else(APPLE)
    include_directories(/afs/andrew.cmu.edu/usr8/efang/private/glm/)
endif (APPLE)

include_directories(${GLFW_INCLUDE_DIRS})
include_directories(${GLEW_INCLUDE_DIRS})
include_directories(${OPENGL_INCLUDE_DIR})

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_52,code=sm_52 -gencode arch=compute_61,code=sm_61)
#set(CUDA_LIBRARY saxpy)
#set(CUDA_SOURCES saxpy.cu)
#cuda_add_library(${CUDA_LIBRARY} ${CUDA_SOURCES})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp glWindow.cpp glWindow.h shader.h glRenderer.cpp glRenderer.h ParticleSystemSerial.cpp
        ParticleSystemSerial.h ParticleSystem.h Camera.h SOIL2.h Particle.h ParticleSystemCUDA.h ParticleSystemCUDA.cpp)

cuda_add_executable(PBF_15418 ${SOURCE_FILES} ParticleSystemCUDA.cu)

if(APPLE)
    target_link_libraries(PBF_15418 ${EXTRA_LIBS})
    set(SOIL2_DIR ${CMAKE_SOURCE_DIR}/lib/libsoil2.a)
else(APPLE)
    set(SOIL2_DIR ${CMAKE_SOURCE_DIR}/lib/linux/libsoil2.a)
endif(APPLE)

target_link_libraries(PBF_15418 ${CUDA_LIBRARIES} ${CUDA_LIBRARY} ${GLFW_LIBRARIES} ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES} ${SOIL2_DIR})
