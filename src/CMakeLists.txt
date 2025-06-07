add_executable(GameOfLife main.cu)

set_target_properties(GameOfLife PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(GameOfLife 
    ${OPENGL_LIBRARIES} 
    glfw 
    glm 
    ${CUDA_LIBRARIES}
)