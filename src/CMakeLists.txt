cmake_minimum_required(VERSION 3.20)

project(GameOfLife CXX CUDA)

add_executable(GameOfLife main.cu)

set_target_properties(GameOfLife PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set_property(TARGET GameOfLife PROPERTY CXX_STANDARD 20)
set_property(TARGET GameOfLife PROPERTY CXX_STANDARD_REQUIRED ON)


target_link_libraries(GameOfLife 
    ${OPENGL_LIBRARIES}
    glfw
    ${GLM_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${OpenCL_LIBRARIES}
)

target_compile_options(GameOfLife
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            $<$<CONFIG:Release>:
                -O3
                --use_fast_math
                -gencode arch=compute_75,code=sm_75
            >
            $<$<CONFIG:Debug>:
                -G
                --lineinfo
            >
        >
)
