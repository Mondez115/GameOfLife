add_executable(GameOfLife main.cu)

set_target_properties(GameOfLife PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET GameOfLife PROPERTY CUDA_ARCHITECTURES 86)
set_property(TARGET GameOfLife PROPERTY CXX_STANDARD 20)
set_property(TARGET GameOfLife PROPERTY CXX_STANDARD_REQUIRED ON)

target_link_libraries(GameOfLife 
    ${OPENGL_LIBRARIES} 
    glfw 
    glm 
    ${CUDA_LIBRARIES}
    ${OpenCL_LIBRARIES}
)