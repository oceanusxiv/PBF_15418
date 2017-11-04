cmake_minimum_required(VERSION 3.9)
if (APPLE)
  set(CMAKE_C_COMPILER /usr/local/Cellar/llvm/5.0.0/bin/clang)
  set(CMAKE_CXX_COMPILER /usr/local/Cellar/llvm/5.0.0/bin/clang++)
endif()
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -Ofast")
# if (CMAKE_BUILD_TYPE EQUAL "DEBUG")
#     set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-no_pie")
# endif()
project(PBF_15418)
option(CUDA "enable CUDA in compilation" OFF)

find_program(
  CLANG_TIDY_EXE
  NAMES "clang-tidy"
  HINTS "/usr/local/Cellar/llvm/5.0.0/bin"
  )
if(NOT CLANG_TIDY_EXE)
  message(STATUS "clang-tidy not found.")
else()
  message(STATUS "clang-tidy found: ${CLANG_TIDY_EXE}")
  set(DO_CLANG_TIDY "${CLANG_TIDY_EXE}" "-checks=*,-clang-analyzer-alpha.*")
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config.h.in ${CMAKE_CURRENT_BINARY_DIR}/config.h)

if (CUDA)
  enable_language(CUDA)
endif()

find_package(GLFW3 REQUIRED)
find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glm REQUIRED)
# find_package(Gperftools REQUIRED)

if (APPLE)
  find_library(COCOA_LIBRARY Cocoa REQUIRED)
endif()

if (CUDA)
  add_executable(PBF_15418
                main.cpp 
                config.h
                glWindow.cpp 
                glWindow.h 
                shader.h 
                glRenderer.cpp 
                glRenderer.h 
                ParticleSystemSerial.cpp
                ParticleSystemSerial.h 
                ParticleSystem.h 
                Camera.h 
                SOIL2.h 
                Particle.h 
                ParticleSystemCUDA.h 
                ParticleSystemCUDA.cpp
                ParticleSystemCUDA.cu)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_52,code=sm_52 -gencode arch=compute_61,code=sm_61 -std=c++11)
else()
  add_executable(PBF_15418
                main.cpp 
                config.h
                glWindow.cpp 
                glWindow.h 
                shader.h 
                glRenderer.cpp 
                glRenderer.h 
                ParticleSystemSerial.cpp
                ParticleSystemSerial.h 
                ParticleSystem.h 
                Camera.h 
                SOIL2.h 
                Particle.h)
endif ()

target_compile_features(PBF_15418 PUBLIC 
  cxx_std_14)

# target_compile_options(PBF_15418 PUBLIC
#   -fopenmp 
#   -Ofast
# )

# target_link_libraries(PBF_15418
#   -fopenmp
#   -Ofast
# )

if(CLANG_TIDY_EXE)
  set_target_properties(
    PBF_15418 PROPERTIES
    CXX_CLANG_TIDY "${DO_CLANG_TIDY}"
  )
endif()

target_include_directories(PBF_15418
  PUBLIC
  ${GLEW_INCLUDE_DIRS}
  ${OPENGL_INCLUDE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR})
  # ${Gperftools_INCLUDE_DIRS}

if(APPLE)
  target_link_libraries(PBF_15418 
  ${COCOA_LIBRARY}
  ${CMAKE_SOURCE_DIR}/lib/libsoil2.a)
else()
  target_link_libraries(PBF_15418 
  ${CMAKE_SOURCE_DIR}/lib/linux/libsoil2.a)
endif()

target_link_libraries(PBF_15418 
  glfw
  ${GLEW_LIBRARIES} 
  ${OPENGL_LIBRARIES})
  # profiler)
