cmake_minimum_required(VERSION 3.3)

project(GameOfLife)

# Find OpenGL
find_package(OpenGL REQUIRED)
include_directories(${OPENGL_INCLUDE_DIRS})

# GLFW settings
set(GLFW_BUILD_DOCS OFF CACHE BOOL "GLFW lib only") 
set(GLFW_INSTALL OFF CACHE BOOL "GLFW lib only")
set(GLAD_GL "")

# Add external directory (assuming it's a submodule or external source directory)
add_subdirectory(extern)

# Set GLM include directory (assuming you have GLM in extern)
set(GLM_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/extern/glm/glm")
include_directories(${GLM_INCLUDE_DIRS})

# Option to use GLFW/GLAD
option(GLFW-CMAKE-STARTER-USE-GLFW-GLAD "Use GLAD from GLFW" ON)
if(GLFW-CMAKE-STARTER-USE-GLFW-GLAD)
    include_directories("${GLFW_SOURCE_DIR}/deps")
    set(GLAD_GL "${GLFW_SOURCE_DIR}/deps/glad/gl.h")
endif()

enable_language(CUDA)

include_directories(${CUDA_INCLUDE_DIRS})

add_subdirectory(src)
