cmake_minimum_required(VERSION 3.20)

project(GameOfLife CXX CUDA)
cmake_policy(SET CMP0072 NEW)

find_package(OpenGL REQUIRED)
include_directories(${OPENGL_INCLUDE_DIRS})

set(GLFW_BUILD_DOCS OFF CACHE BOOL "GLFW lib only") 
set(GLFW_INSTALL    OFF CACHE BOOL "GLFW lib only")
set(GLAD_GL         "")

find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})

if(MSVC)
    set(GLM_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/extern/glm/glm")
    include_directories(${GLM_INCLUDE_DIRS})
else()
    find_package(glm REQUIRED)
    include_directories(${GLM_INCLUDE_DIRS})
endif()

find_package(glfw3 REQUIRED)
include_directories(${GLFW_INCLUDE_DIRS})

set(CMAKE_CUDA_ARCHITECTURES 75)

enable_language(CUDA)
include_directories(${CUDA_INCLUDE_DIRS})

set(CMAKE_CUDA_FLAGS_RELEASE
    "${CMAKE_CUDA_FLAGS_RELEASE}
     -O3                      # optimización de nivel 3
     --use_fast_math          # matemática acelerada (imprecisa)
     -gencode arch=compute_75,code=sm_75  # target SM 7.5
    ")

set(CMAKE_CUDA_FLAGS_DEBUG
    "${CMAKE_CUDA_FLAGS_DEBUG}
     -G       # genera debug info en device
     --lineinfo
    ")

add_subdirectory(src)
