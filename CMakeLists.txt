cmake_minimum_required(VERSION 3.20)

project(GameOfLife CXX)
cmake_policy(SET CMP0072 NEW)
# Find OpenGL
find_package(OpenGL REQUIRED)
include_directories(${OPENGL_INCLUDE_DIRS})

# GLFW settings
set(GLFW_BUILD_DOCS OFF CACHE BOOL "GLFW lib only") 
set(GLFW_INSTALL OFF CACHE BOOL "GLFW lib only")
set(GLAD_GL "")

# find OpenCL
find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})

# Add external directory (assuming it's a submodule or external source directory)
# add_subdirectory(extern)

# Set GLM include directory
if(MSVC)
    set(GLM_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/extern/glm/glm")
    include_directories(${GLM_INCLUDE_DIRS})
else()
    find_package(glm REQUIRED)
    include_directories(${GLM_INCLUDE_DIRS})
endif()

# Option to use GLFW/GLAD
find_package(glfw3 REQUIRED)
include_directories(${GLFW_INCLUDE_DIRS})
# option(GLFW-CMAKE-STARTER-USE-GLFW-GLAD "Use GLAD from GLFW" ON)
# if(GLFW-CMAKE-STARTER-USE-GLFW-GLAD)
#     include_directories("${GLFW_SOURCE_DIR}/deps")
#     set(GLAD_GL "${GLFW_SOURCE_DIR}/deps/glad/gl.h")
# endif()

set(CMAKE_CUDA_ARCHITECTURES 86)
enable_language(CUDA)

include_directories(${CUDA_INCLUDE_DIRS})

add_subdirectory(src)
