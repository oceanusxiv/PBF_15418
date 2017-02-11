cmake_minimum_required(VERSION 3.6)
project(PBF_15418)

find_package(PkgConfig REQUIRED)
pkg_search_module(GLFW REQUIRED glfw3)
find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)

IF(APPLE)
    INCLUDE_DIRECTORIES ( /System/Library/Frameworks )
    FIND_LIBRARY(COCOA_LIBRARY Cocoa)
    MARK_AS_ADVANCED (COCOA_LIBRARY)
    SET(EXTRA_LIBS ${COCOA_LIBRARY})
ENDIF (APPLE)

include_directories(${GLFW_INCLUDE_DIRS})
include_directories(${GLEW_INCLUDE_DIRS})
include_directories(${OPENGL_INCLUDE_DIR})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp glWindow.cpp glWindow.h shader.h glRenderer.cpp glRenderer.h particleSystem.cpp particleSystem.h Camera.h SOIL2.h)

add_executable(PBF_15418 ${SOURCE_FILES})

if(APPLE)
    target_link_libraries(PBF_15418 ${EXTRA_LIBS})
endif(APPLE)

target_link_libraries(PBF_15418 ${GLFW_LIBRARIES} ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES} ${CMAKE_SOURCE_DIR}/lib/libSOIL2.a)