
#include <GLFW/glfw3.h>
#include <GL/gl.h>
#include <stdio.h>
#include <vector>
#include <iostream>
#include <sstream> 
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <cmath>
#include <chrono>
#include <random>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <iomanip>
#include <string>
#include <csignal>

int width = 1024;
int height = 1024;

int GRID_X, GRID_Y;
bool double_dim;

int block_size;

std::string method;

std::string CUDA = "CUDA";
std::string CPU = "CPU";

std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
std::chrono::high_resolution_clock::time_point stop = std::chrono::high_resolution_clock::now();

std::chrono::high_resolution_clock::time_point frame_start = std::chrono::high_resolution_clock::now();
std::chrono::high_resolution_clock::time_point frame_end = std::chrono::high_resolution_clock::now(); 

bool running = true;

int num_elements, grid_size;

std::vector<char> cells, next_cells;
std::vector<std::vector<char>> cells_double, next_cells_double;

char* d_cells, *d_next_cells, *d_cells_double, *d_next_cells_double;

cudaError_t err;

float zoom_factor = 1.0f;

void signal_handler(int signal) {
    if (signal == SIGINT) {
        std::cout << "\n Stopping program..." << std::endl;
        running = false;
    }
}

void print_config(){

    using namespace std;

    cout << "Grid Columns: " << GRID_X << endl;
    cout << "Grid Rows: " << GRID_Y << endl;
    cout << "Method: " << method << endl;
    
    if (method == CUDA){
        cout << "Double dimension: " << (double_dim ? "true" : "false") << endl;
        cout << "Block size: " << block_size << endl;
    }

}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
 
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        running = false;
        std::cout << "\nESC key pressed. Exiting loop." << std::endl;
    }
}

void init_vectors() {
     
    cells = std::vector<char>(GRID_X * GRID_Y, 0);
    next_cells = std::vector<char>(GRID_X * GRID_Y, 0);

    if (method == CPU) {
        
        // Random number generator for initializing cells
        std::random_device rd;
        std::mt19937 gen(rd());  // Mersenne Twister random number generator
        std::uniform_int_distribution<> dis(0, 1);  // Random values between 0 and 2 (inclusive)

        for (int i = 0; i < GRID_X * GRID_Y; ++i) {
            int rand_value = dis(gen);  // Random value in {0, 1, 2}
            cells[i] = (rand_value < 0.1);  // If rand_value < 3, set cell to true (about 33% chance)
        }

        std::cout << "CPU: Initialized cells and next_cells with random values." << std::endl;
    }
}


int get_neighbours(int x, int y) {
    x = (x + GRID_X) % GRID_X;
    y = (y + GRID_Y) % GRID_Y;

    return cells[(x + 1) % GRID_X + y * GRID_X] +
           cells[(x - 1 + GRID_X) % GRID_X + y * GRID_X] +
           cells[x + ((y + 1) % GRID_Y) * GRID_X] +
           cells[x + ((y - 1 + GRID_Y) % GRID_Y) * GRID_X] +
           cells[(x + 1) % GRID_X + ((y + 1) % GRID_Y) * GRID_X] +
           cells[(x - 1 + GRID_X) % GRID_X + ((y + 1) % GRID_Y) * GRID_X] +
           cells[(x + 1) % GRID_X + ((y - 1 + GRID_Y) % GRID_Y) * GRID_X] +
           cells[(x - 1 + GRID_X) % GRID_X + ((y - 1 + GRID_Y) % GRID_Y) * GRID_X];
}

void game_of_life_cpu() {
    for (int j = 0; j < GRID_Y; j++) {
        for (int i = 0; i < GRID_X; i++) {
            int neighbors = get_neighbours(i, j);
            int idx = j * GRID_X + i;

            if (cells[idx]) {
                next_cells[idx] = (neighbors == 2 || neighbors == 3);
            } else {
                next_cells[idx] = (neighbors == 3);
            }
        }
    }
    std::swap(cells, next_cells);
}

void draw_grid(int rows, int cols) {

    glBegin(GL_POINTS);

    for (int j = 0; j < rows; j++) {
        for (int i = 0; i < cols; i++) {
            
                float x = i - cols / 2;
                float y = -j + rows / 2;

                if (cells[i + j * cols] == 1) {
                    glColor3f(1.0f, 0.0f, 0.0f);
                } else {
                    glColor3f(0.0f, 0.0f, 0.0f);
                }

                glVertex2f(x, y);
            
        }
    }

    glEnd();

}

void initialize_camera(int rows, int cols, int window_width, int window_height) {

    float zoom_factor = static_cast<float>(std::max(GRID_X*1.0/window_width, GRID_Y*1.0/window_height));

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    float zoomed_width = window_width * zoom_factor;
    float zoomed_height = window_height * zoom_factor;

    glOrtho(-zoomed_width / 2, zoomed_width / 2, -zoomed_height / 2, zoomed_height / 2, -1.0, 1.0);

    glMatrixMode(GL_MODELVIEW);
}

int setup_cuda_memory(int grid_x, int grid_y){

    err = cudaMalloc((void**)&d_cells, GRID_X * GRID_Y * sizeof(char));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed for d_cells: " << cudaGetErrorString(err) << std::endl;
        return -9;
    }

    err = cudaMalloc((void**)&d_next_cells, GRID_X * GRID_Y * sizeof(char));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed for d_next_cells: " << cudaGetErrorString(err) << std::endl;
        return -10;
    }

    return 0;
}

GLFWwindow* init_glfw(int width, int height){

    if (!glfwInit()) {
        return NULL;
    }

    GLFWwindow* window = glfwCreateWindow(width, height, "Game Of Life", NULL, NULL);

    if (!window) {
        glfwTerminate();
        return NULL;
    }

    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    glfwSetKeyCallback(window, key_callback);
    glfwMakeContextCurrent(window);
    glViewport(0, 0, width, height);
    glfwSwapInterval(1);
    glfwSetWindowAttrib(window, GLFW_RESIZABLE, GLFW_FALSE);


    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glLineWidth(1.0f);

    glClear(GL_COLOR_BUFFER_BIT); 

    return window;
}

__global__ void game_of_life_kernel(char* d_cells, char* d_next_cells, int grid_x, int grid_y) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < grid_x * grid_y) {

        int x = idx % grid_x;
        int y = idx / grid_x;

        int neighbors = 0;

        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0) continue;

                int nx = (x + dx + grid_x) % grid_x;
                int ny = (y + dy + grid_y) % grid_y;

                neighbors += d_cells[nx + ny * grid_x];
            }
        }

        if (d_cells[idx]) {
            d_next_cells[idx] = (neighbors == 2 || neighbors == 3);
        } else {
            d_next_cells[idx] = (neighbors == 3);
        }
    }
}

__global__ void game_of_life_kernel_2d(char* d_cells, char* d_next_cells, int grid_x, int grid_y) {

    int block_x = blockIdx.x * blockDim.x + threadIdx.x;
    int block_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (block_x < grid_x && block_y < grid_y) {

        int neighbors = 0;

        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {

                if (dx == 0 && dy == 0) continue;

                int nx = (block_x + dx + grid_x) % grid_x;
                int ny = (block_y + dy + grid_y) % grid_y;

                neighbors += d_cells[nx + ny * grid_x];
            }
        }

        int idx = block_x + block_y * grid_x;
        if (d_cells[idx]) {
            d_next_cells[idx] = (neighbors == 2 || neighbors == 3);
        } else {
            d_next_cells[idx] = (neighbors == 3);
        }
    }
}

__global__ void randomize_grid_cuda_kernel(char* d_cells, int grid_x, int grid_y, unsigned long long seed) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < grid_x * grid_y) {
     
        int x = idx % grid_x;
        int y = idx / grid_x;

        curandState state;
        curand_init(seed, idx, 0, &state);

        float random_value = curand_uniform(&state);

        if (random_value < 0.1f) {
            d_cells[idx] = 1;
        } else {
            d_cells[idx] = 0;
        }
    }
}


int main(int argc, char** argv) {

    std::signal(SIGINT, signal_handler);

    if(!(argc == 7)){
        return -2;
    }

    int draw = std::atoi(argv[1]);

    GRID_X = std::atoi(argv[2]);

    if (GRID_X <= 0){
        return -3;
    }

    GRID_Y = std::atoi(argv[3]);

    if (GRID_Y <= 0){
        return -4;
    }

    method = argv[4];

 

    double_dim = std::atoi(argv[5]);
    block_size = std::atoi(argv[6]);

    init_vectors();

    dim3 block_size_2d(2, block_size);
    dim3 grid_size_2d((GRID_X + block_size_2d.x - 1) / block_size_2d.x,
                  (GRID_Y + block_size_2d.y - 1) / block_size_2d.y);

    if(method == CUDA){
        num_elements = GRID_X * GRID_Y;
        grid_size = (num_elements + block_size - 1) / block_size;


        if(setup_cuda_memory(GRID_X, GRID_Y)){
            return -11;
        }

        cudaMemcpy(d_cells, cells.data(), GRID_X * GRID_Y * sizeof(char), cudaMemcpyHostToDevice);
        cudaMemcpy(d_next_cells, next_cells.data(), GRID_X * GRID_Y * sizeof(char), cudaMemcpyHostToDevice);

        randomize_grid_cuda_kernel<<<grid_size, block_size>>>(d_cells,GRID_X, GRID_Y, 1);
    }

    print_config();

    unsigned long long cells_proccesed = 0;

    std::chrono::high_resolution_clock::time_point app_start = std::chrono::high_resolution_clock::now();

    std::chrono::high_resolution_clock::time_point app_end;

    if(draw){
        if (!glfwInit()) {
            fprintf(stderr, "Failed to initialize GLFW\n");
            return -1;
        }

        GLFWwindow* window = init_glfw(width, height);

        if (!window) {
            return -8;
        } 

        initialize_camera(GRID_X,GRID_Y,width,height); 

        while (!glfwWindowShouldClose(window) && running) {

            frame_start = std::chrono::high_resolution_clock::now();

            if (method == CPU) {
        

                game_of_life_cpu();


            } else {

                start = std::chrono::high_resolution_clock::now();

                if(double_dim){
                    game_of_life_kernel_2d<<<grid_size_2d, block_size_2d>>>(d_cells, d_next_cells, GRID_X, GRID_Y);
                } else {
                    game_of_life_kernel<<<grid_size, block_size>>>(d_cells, d_next_cells, GRID_X, GRID_Y);
                }

                cudaDeviceSynchronize();
        
                err = cudaGetLastError();
                if (err != cudaSuccess) {
                    std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;
                    break;
                }

                cudaMemcpy(cells.data(), d_next_cells, GRID_X * GRID_Y * sizeof(char), cudaMemcpyDeviceToHost);
                cudaMemcpy(next_cells.data(), d_cells, GRID_X * GRID_Y * sizeof(char), cudaMemcpyDeviceToHost);

                stop = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        
                char* temp = d_cells;
                d_cells = d_next_cells;
                d_next_cells = temp;

                        
                stop = std::chrono::high_resolution_clock::now();
                duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    
            }
        
            draw_grid(GRID_X, GRID_Y);

            glfwSwapBuffers(window);
            glfwPollEvents();

            frame_end = std::chrono::high_resolution_clock::now();

            app_end = frame_end;
            cells_proccesed+= GRID_X*GRID_Y; 

            auto time_between_frames = std::chrono::duration_cast<std::chrono::microseconds>(frame_end - frame_start);

            auto time_app = std::chrono::duration_cast<std::chrono::microseconds>(app_end - app_start);

            double cells_per_second = cells_proccesed/(time_app.count()/1000000.f);

            int count = time_between_frames.count();

            if (count == 0){
                count = 1000000;
            }


            int fps = 1000000/count;

            std::ostringstream titleStream;
            titleStream << "GameOfLife - FPS: " << fps << " Cells per second: " << cells_per_second;

            glfwSetWindowTitle(window, titleStream.str().c_str());
        }

        glfwDestroyWindow(window);
        glfwTerminate();
    } else { 

        while(running){

        if (method == CPU) {
        
                game_of_life_cpu();
                app_end = std::chrono::high_resolution_clock::now();

            } else {


                if(double_dim){
                    game_of_life_kernel_2d<<<grid_size_2d, block_size_2d>>>(d_cells, d_next_cells, GRID_X, GRID_Y);
                } else {
                    game_of_life_kernel<<<grid_size, block_size>>>(d_cells, d_next_cells, GRID_X, GRID_Y);
                }

                err = cudaGetLastError();
                if (err != cudaSuccess) {
                    std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;
                    return;
                }

                cudaDeviceSynchronize();

                cudaMemcpy(cells.data(), d_next_cells, GRID_X * GRID_Y * sizeof(char), cudaMemcpyDeviceToHost);
                app_end = std::chrono::high_resolution_clock::now();

                char* temp = d_cells;
                d_cells = d_next_cells;
                d_next_cells = temp;
            }
        
            cells_proccesed+=GRID_X*GRID_Y; 

            auto time_app = std::chrono::duration_cast<std::chrono::microseconds>(app_end - app_start);

            double cells_per_second = cells_proccesed/(time_app.count()/1000000.f);

            std::ostringstream titleStream;
    
            std::cout << "\rCells per second: " << static_cast<unsigned long long>(cells_per_second) << std::flush;

        }
    }

    std::cout << "\nMain loop ended" << std::endl;


    if(method == CUDA){
        cudaFree(d_cells);
        cudaFree(d_next_cells);

        std::cout << "Freed CUDA resources" << std::endl;
    }    

    return 0;
}
