#include <CL/cl.h>
#include <GL/gl.h>
#include <GLFW/glfw3.h>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdio.h>
#include <string>
#include <vector>

int width = 1024;
int height = 1024;

int GRID_X, GRID_Y;
bool double_dim;

int block_size;
unsigned long long seed =
    std::chrono::system_clock::now().time_since_epoch().count();
std::string method;

std::string CUDA = "CUDA";
std::string CPU = "CPU";
std::string OPENCL = "OPENCL";

std::chrono::high_resolution_clock::time_point start =
    std::chrono::high_resolution_clock::now();
std::chrono::high_resolution_clock::time_point stop =
    std::chrono::high_resolution_clock::now();

std::chrono::high_resolution_clock::time_point frame_start =
    std::chrono::high_resolution_clock::now();
std::chrono::high_resolution_clock::time_point frame_end =
    std::chrono::high_resolution_clock::now();

bool running = true;

int num_elements, grid_size;

std::vector<char> cells, next_cells;
std::vector<std::vector<char>> cells_double, next_cells_double;

char *d_cells, *d_next_cells, *d_cells_double, *d_next_cells_double;

cl_uint work_dim;
size_t global_work_size;
size_t local_work_size;
size_t global_work_size_2d[2];
size_t local_work_size_2d[2];

cudaError_t err_cuda;
cudaEvent_t ev_start, ev_stop;


cl_int err_cl;
cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue queue;
cl_program program;

cl_mem opencl_d_cells, opencl_d_next_cells;
cl_kernel randomize_kernel, grid_1d_kernel, grid_2d_kernel;
cl_program randomize_program, grid_1d_program, grid_2d_program;

float zoom_factor = 1.0f;

void signal_handler(int signal) {
  if (signal == SIGINT) {
    std::cout << "\n Stopping program..." << std::endl;
    running = false;
  }
}

void print_config() {

  using namespace std;

  cout << "Grid Columns: " << GRID_X << endl;
  cout << "Grid Rows: " << GRID_Y << endl;
  cout << "Method: " << method << endl;

  if (method == CUDA) {
    cout << "Double dimension: " << (double_dim ? "true" : "false") << endl;
    cout << "Block size: " << block_size << endl;
  } else if (method == OPENCL) {
    cout << "Double dimension: " << (double_dim ? "true" : "false") << endl;
    size_t device_name_size;
    clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &device_name_size);
    char *device_name = (char *)malloc(device_name_size);
    clGetDeviceInfo(device, CL_DEVICE_NAME, device_name_size, device_name,
                    NULL);
    cout << "Device: " << device_name << endl;
    if (double_dim) {
      cout << "Global work size: " << global_work_size_2d[0] << " x "
           << global_work_size_2d[1] << endl;
      cout << "Local work size: " << local_work_size_2d[0] << " x "
           << local_work_size_2d[1] << endl;
      cout << "Total work items: "
           << (global_work_size_2d[0] * global_work_size_2d[1]) << endl;
      cout << "Required work items: " << (GRID_X * GRID_Y) << endl;
    } else {
      cout << "Global work size: " << global_work_size << endl;
      cout << "Local work size: " << local_work_size << endl;
    }
  }
}

void key_callback(GLFWwindow *window, int key, int scancode, int action,
                  int mods) {

  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
    running = false;
    std::cout << "\nESC key pressed. Exiting loop." << std::endl;
  }
}

void init_vectors() {

  cells = std::vector<char>(GRID_X * GRID_Y, 0);
  next_cells = std::vector<char>(GRID_X * GRID_Y, 0);

  if (method != CUDA) {

    // Random number generator for initializing cells
    std::random_device rd;
    std::mt19937 gen(rd()); // Mersenne Twister random number generator
    std::uniform_int_distribution<> dis(
        0, 1); // Random values between 0 and 2 (inclusive)

    for (int i = 0; i < GRID_X * GRID_Y; ++i) {
      int rand_value = dis(gen); // Random value in {0, 1, 2}
      cells[i] =
          (rand_value <
           0.1); // If rand_value < 3, set cell to true (about 33% chance)
    }

    std::cout << "CPU: Initialized cells and next_cells with random values."
              << std::endl;
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
         cells[(x - 1 + GRID_X) % GRID_X +
               ((y - 1 + GRID_Y) % GRID_Y) * GRID_X];
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
        glColor3f(0.0f, 1.0f, 0.0f);
      } else {
        glColor3f(0.0f, 0.0f, 0.0f);
      }

      glVertex2f(x, y);
    }
  }

  glEnd();
}

void initialize_camera(int rows, int cols, int window_width,
                       int window_height) {

  float zoom_factor = static_cast<float>(
      std::max(GRID_X * 1.0 / window_width, GRID_Y * 1.0 / window_height));

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();

  float zoomed_width = window_width * zoom_factor;
  float zoomed_height = window_height * zoom_factor;

  glOrtho(-zoomed_width / 2, zoomed_width / 2, -zoomed_height / 2,
          zoomed_height / 2, -1.0, 1.0);

  glMatrixMode(GL_MODELVIEW);
}

int setup_cuda_memory(int grid_x, int grid_y) {

  err_cuda = cudaMalloc((void **)&d_cells, GRID_X * GRID_Y * sizeof(char));
  if (err_cuda != cudaSuccess) {
    std::cerr << "CUDA malloc failed for d_cells: "
              << cudaGetErrorString(err_cuda) << std::endl;
    return -9;
  }

  err_cuda = cudaMalloc((void **)&d_next_cells, GRID_X * GRID_Y * sizeof(char));
  if (err_cuda != cudaSuccess) {
    std::cerr << "CUDA malloc failed for d_next_cells: "
              << cudaGetErrorString(err_cuda) << std::endl;
    return -10;
  }

  return 0;
}

GLFWwindow *init_glfw(int width, int height) {

  if (!glfwInit()) {
    return NULL;
  }

  GLFWwindow *window =
      glfwCreateWindow(width, height, "Game Of Life", NULL, NULL);

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

__global__ void game_of_life_kernel(const char* __restrict__ d_cells, char *d_next_cells,
                                    int grid_x, int grid_y) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < grid_x * grid_y) {

    int x = idx % grid_x;
    int y = idx / grid_x;

    int neighbors = 0;

    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        if (dx == 0 && dy == 0)
          continue;

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

__global__ void game_of_life_kernel_2d(char *d_cells, char *d_next_cells,
                                       int grid_x, int grid_y) {

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < grid_x && y < grid_y) {

    int neighbors = 0;

    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {

        if (dx == 0 && dy == 0)
          continue;

        int nx = (x + dx + grid_x) % grid_x;
        int ny = (y + dy + grid_y) % grid_y;

        neighbors += __ldg(d_cells + nx + ny * grid_x);
      }
    }

    int idx = x + y * grid_x;
    if (d_cells[idx]) {
      d_next_cells[idx] = (neighbors == 2 || neighbors == 3);
    } else {
      d_next_cells[idx] = (neighbors == 3);
    }
  }
}

__global__ void randomize_grid_cuda_kernel(char *d_cells, int grid_x,
                                           int grid_y,
                                           unsigned long long seed) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < grid_x * grid_y) {

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

const char *game_of_life_kernel_source =
    "__kernel void game_of_life_kernel_opencl( \n"
    "    __global const char* d_cells,\n"      // current state array
    "    __global char*       d_next_cells,\n" // next state array
    "    const int            grid_x,\n"       // number of columns
    "    const int            grid_y\n"        // number of rows
    ") { \n"
    // Compute 1D global index for this work-item \n
    "    int idx = get_global_id(0); \n"

    // Total number of cells \n
    "    int total = grid_x * grid_y; \n"

    // Bounds check: exit if beyond the grid \n
    "    if (idx >= total) return; \n"

    // Compute 2D coordinates from 1D index \n
    "    int x = idx % grid_x; \n"
    "    int y = idx / grid_x; \n"

    // Count live neighbors with toroidal wrapping \n
    "    int neighbors = 0; \n"
    "    for (int dy = -1; dy <= 1; ++dy) { \n"
    "        for (int dx = -1; dx <= 1; ++dx) { \n"
    "            if (dx == 0 && dy == 0) continue; \n"
    "            int nx = (x + dx + grid_x) % grid_x; \n"
    "            int ny = (y + dy + grid_y) % grid_y; \n"
    "            neighbors += d_cells[nx + ny * grid_x]; \n"
    "        } \n"
    "    } \n"

    // Apply Conway's rules \n
    "    if (d_cells[idx]) { \n"
    "        d_next_cells[idx] = (neighbors == 2 || neighbors == 3); \n"
    "    } else { \n"
    "        d_next_cells[idx] = (neighbors == 3); \n"
    "    } \n"
    "}";

const char *game_of_life_kernel_2d_source =
    "__kernel void game_of_life_kernel_2d_opencl(\n"
    "    __global const char* d_cells,\n"
    "    __global       char* d_next_cells,\n"
    "    const int      grid_x,\n"
    "    const int      grid_y\n"
    ") {\n"
    "    int x = get_global_id(0);\n"
    "    int y = get_global_id(1);\n"
    "\n"
    "    if (x < grid_x && y < grid_y) {\n"
    "        int neighbors = 0;\n"
    "\n"
    "        for (int dy = -1; dy <= 1; ++dy) {\n"
    "            for (int dx = -1; dx <= 1; ++dx) {\n"
    "                if (dx == 0 && dy == 0) continue;\n"
    "\n"
    "                int nx = (x + dx + grid_x) % grid_x;\n"
    "                int ny = (y + dy + grid_y) % grid_y;\n"
    "\n"
    "                neighbors += d_cells[nx + ny * grid_x];\n"
    "            }\n"
    "        }\n"
    "\n"
    "        int idx = x + y * grid_x;\n"
    "        char current = d_cells[idx];\n"
    "        if (current) {\n"
    "            d_next_cells[idx] = (neighbors == 2 || neighbors == 3) ? 1 : "
    "0;\n"
    "        } else {\n"
    "            d_next_cells[idx] = (neighbors == 3) ? 1 : 0;\n"
    "        }\n"
    "    }\n"
    "}\n";

int getCudaDeviceCount(int &deviceCount) {
  cudaError_t err = cudaGetDeviceCount(&deviceCount);
  if (err != cudaSuccess) {
    std::cerr << "Error: Failed to get CUDA device count: "
              << cudaGetErrorString(err) << std::endl;
    return 1;
  }

  if (deviceCount == 0) {
    std::cerr << "Error: No CUDA-compatible devices found." << std::endl;
    return 1;
  }

  return 0;
}

int selectCudaDevice(int device) {
  cudaError_t err = cudaSetDevice(device);
  if (err != cudaSuccess) {
    std::cerr << "Error: Failed to set CUDA device " << device << ": "
              << cudaGetErrorString(err) << std::endl;
    return 1;
  }

  cudaDeviceProp deviceProp;
  err = cudaGetDeviceProperties(&deviceProp, device);
  if (err != cudaSuccess) {
    std::cerr << "Error: Failed to get properties for device " << device << ": "
              << cudaGetErrorString(err) << std::endl;
    return 1;
  }

  std::cout << "Using device " << device << ": " << deviceProp.name
            << std::endl;
  return 0;
}

int main(int argc, char **argv) {

  std::signal(SIGINT, signal_handler);

  if (!(argc == 7)) {
    return -2;
  }

  int draw = std::atoi(argv[1]);

  GRID_X = std::atoi(argv[2]);

  if (GRID_X <= 0) {
    return -3;
  }

  GRID_Y = std::atoi(argv[3]);

  if (GRID_Y <= 0) {
    return -4;
  }

  method = argv[4];

  double_dim = std::atoi(argv[5]);
  block_size = std::atoi(argv[6]);

  init_vectors();

  dim3 block_size_2d(block_size, block_size);
  dim3 grid_size_2d((GRID_X + block_size_2d.x - 1) / block_size_2d.x,
                    (GRID_Y + block_size_2d.y - 1) / block_size_2d.y);

  const size_t local_work_size_2d[2] = {
      (size_t)block_size, // x dimension of a work-group
      (size_t)block_size  // y dimension of a work-group
  };

  size_t num_groups_x = (GRID_X + block_size - 1) / block_size;
  size_t num_groups_y = (GRID_Y + block_size - 1) / block_size;

  const size_t global_work_size_2d[2] = {
      num_groups_x * local_work_size_2d[0], // covers all GRID_X columns
      num_groups_y * local_work_size_2d[1]  // covers all GRID_Y rows
  };

  const size_t local_work_size[1] = {block_size};

  const size_t total_cells = GRID_X * GRID_Y;
  const size_t global_work_size[1] = {
      ((total_cells + block_size - 1) / block_size) * block_size};

  if (method == CUDA) {

    int device_count = 0;

    if (getCudaDeviceCount(device_count)) {
      return;
    }

    if (selectCudaDevice(0)) {
      return;
    }

    num_elements = GRID_X * GRID_Y;
    grid_size = (num_elements + block_size - 1) / block_size;

    if (setup_cuda_memory(GRID_X, GRID_Y)) {
      return -11;
    }

    cudaMemcpy(d_cells, cells.data(), GRID_X * GRID_Y * sizeof(char),
               cudaMemcpyHostToDevice);

    randomize_grid_cuda_kernel<<<grid_size, block_size>>>(d_cells, GRID_X,
                                                          GRID_Y, seed);
  } else if (method == OPENCL) {

    cl_uint numPlatforms = 0;
    clGetPlatformIDs(0, NULL, &numPlatforms);
    cl_platform_id *platforms =
        (cl_platform_id *)malloc(sizeof(cl_platform_id) * numPlatforms);
    clGetPlatformIDs(numPlatforms, platforms, NULL);

    platform = platforms[1]; // choose the first platform
    free(platforms);

    cl_uint numDevices = 0;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
    cl_device_id *devices =
        (cl_device_id *)malloc(sizeof(cl_device_id) * numDevices);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);

    device = devices[0]; // choose the first device
    free(devices);

    // Query device capabilities to avoid CL_INVALID_WORK_GROUP_SIZE
    size_t max_work_group_size;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t),
                    &max_work_group_size, NULL);

    std::cout << "OpenCL Device max work group size: " << max_work_group_size
              << std::endl;

    size_t max_work_item_dimensions;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(size_t),
                    &max_work_item_dimensions, NULL);

    std::cout << "OpenCL Device max work item dimensions: "
              << max_work_item_dimensions << std::endl;

    size_t max_work_item_sizes[3];
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES,
                    sizeof(max_work_item_sizes), max_work_item_sizes, NULL);

    std::cout << "OpenCL Device max work group size: " << max_work_group_size
              << std::endl;
    std::cout << "OpenCL Device max work item sizes: ["
              << max_work_item_sizes[0] << ", " << max_work_item_sizes[1]
              << ", " << max_work_item_sizes[2] << "]" << std::endl;

    // Adjust work sizes based on device capabilities
    size_t adjusted_block_size = block_size;

    // For 1D: ensure local work size doesn't exceed max work group size
    if (adjusted_block_size > max_work_group_size) {
      adjusted_block_size = max_work_group_size;
      std::cout << "Warning: Adjusted 1D block size from " << block_size
                << " to " << adjusted_block_size << std::endl;
    }

    // For 2D: ensure each dimension and total don't exceed limits
    size_t adjusted_block_size_2d = block_size;
    if (adjusted_block_size_2d > max_work_item_sizes[0] ||
        adjusted_block_size_2d > max_work_item_sizes[1]) {
      adjusted_block_size_2d =
          std::min(max_work_item_sizes[0], max_work_item_sizes[1]);
      std::cout << "Warning: Adjusted 2D block size per dimension from "
                << block_size << " to " << adjusted_block_size_2d << std::endl;
    }

    // Check total work items in 2D work group
    if (adjusted_block_size_2d * adjusted_block_size_2d > max_work_group_size) {
      adjusted_block_size_2d = 16;
      std::cout << "Warning: Adjusted 2D block size for total work group from "
                << block_size << " to " << adjusted_block_size_2d << std::endl;
    }

    // Recalculate work sizes with adjusted values
    const size_t adjusted_local_work_size[1] = {adjusted_block_size};
    const size_t adjusted_local_work_size_2d[2] = {adjusted_block_size_2d,
                                                   adjusted_block_size_2d};

    const size_t adjusted_total_cells = GRID_X * GRID_Y;
    const size_t adjusted_global_work_size[1] = {
        ((adjusted_total_cells + adjusted_block_size - 1) /
         adjusted_block_size) *
        adjusted_block_size};

    size_t adjusted_num_groups_x =
        (GRID_X + adjusted_block_size_2d - 1) / adjusted_block_size_2d;
    size_t adjusted_num_groups_y =
        (GRID_Y + adjusted_block_size_2d - 1) / adjusted_block_size_2d;

    const size_t adjusted_global_work_size_2d[2] = {
        adjusted_num_groups_x * adjusted_local_work_size_2d[0],
        adjusted_num_groups_y * adjusted_local_work_size_2d[1]};

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err_cl);

    // 3. Create command queue and memory buffers
    queue = clCreateCommandQueue(context, device, 0, &err_cl);

    // Create buffers without initial data - fix the initialization
    opencl_d_cells =
        clCreateBuffer(context, CL_MEM_READ_WRITE,
                       GRID_X * GRID_Y * sizeof(char), NULL, &err_cl);
    if (err_cl != CL_SUCCESS) {
      std::cerr << "OpenCL: Failed to create d_cells buffer, error: " << err_cl
                << std::endl;
      return -12;
    }

    opencl_d_next_cells =
        clCreateBuffer(context, CL_MEM_READ_WRITE,
                       GRID_X * GRID_Y * sizeof(char), NULL, &err_cl);
    if (err_cl != CL_SUCCESS) {
      std::cerr << "OpenCL: Failed to create d_next_cells buffer, error: "
                << err_cl << std::endl;
      return -13;
    }

    // Explicitly write initial random data to the device buffer
    err_cl = clEnqueueWriteBuffer(queue, opencl_d_cells, CL_TRUE, 0,
                                  GRID_X * GRID_Y * sizeof(char), cells.data(),
                                  0, NULL, NULL);
    if (err_cl != CL_SUCCESS) {
      std::cerr << "OpenCL: Failed to write initial data to buffer, error: "
                << err_cl << std::endl;
      return -14;
    }

    // 5. Build grid_1d program and kernel
    grid_1d_program = clCreateProgramWithSource(
        context, 1, &game_of_life_kernel_source, NULL, &err_cl);
    err_cl = clBuildProgram(grid_1d_program, 1, &device, NULL, NULL, NULL);
    grid_1d_kernel =
        clCreateKernel(grid_1d_program, "game_of_life_kernel_opencl", &err_cl);

    // 6. Build grid_2d program and kernel
    grid_2d_program = clCreateProgramWithSource(
        context, 1, &game_of_life_kernel_2d_source, NULL, &err_cl);
    err_cl = clBuildProgram(grid_2d_program, 1, &device, NULL, NULL, NULL);
    grid_2d_kernel = clCreateKernel(grid_2d_program,
                                    "game_of_life_kernel_2d_opencl", &err_cl);

    ::global_work_size = adjusted_global_work_size[0];
    ::local_work_size = adjusted_local_work_size[0];
    ::global_work_size_2d[0] = adjusted_global_work_size_2d[0];
    ::global_work_size_2d[1] = adjusted_global_work_size_2d[1];
    ::local_work_size_2d[0] = adjusted_local_work_size_2d[0];
    ::local_work_size_2d[1] = adjusted_local_work_size_2d[1];
  }

  print_config();

  unsigned long long cells_proccesed = 0;

  std::chrono::high_resolution_clock::time_point app_start =
      std::chrono::high_resolution_clock::now();

  std::chrono::high_resolution_clock::time_point app_end;

  if (draw) {
    if (!glfwInit()) {
      fprintf(stderr, "Failed to initialize GLFW\n");
      return -1;
    }

    GLFWwindow *window = init_glfw(width, height);

    if (!window) {
      return -8;
    }

    initialize_camera(GRID_X, GRID_Y, width, height);

    while (!glfwWindowShouldClose(window) && running) {

      frame_start = std::chrono::high_resolution_clock::now();

      if (method == CPU) {

        game_of_life_cpu();

      } else if (method == CUDA) {
        cudaEventCreate(&ev_start);
        cudaEventCreate(&ev_stop);
        cudaEventRecord(ev_start, 0);
        if (double_dim) {
          game_of_life_kernel_2d<<<grid_size_2d, block_size_2d, 0, 0>>>(
              d_cells, d_next_cells, GRID_X, GRID_Y);
        } else {
          game_of_life_kernel<<<grid_size, block_size, 0, 0>>>(
              d_cells, d_next_cells, GRID_X, GRID_Y);
        }
        
        err_cuda = cudaGetLastError();
        if (err_cuda != cudaSuccess) {
          std::cerr << "CUDA kernel launch failed: "
          << cudaGetErrorString(err_cuda) << std::endl;
          return -1;
        }
        
        cudaMemcpy(cells.data(), d_next_cells, GRID_X * GRID_Y * sizeof(char),
        cudaMemcpyDeviceToHost);

        cudaEventRecord(ev_stop, 0);
        cudaEventSynchronize(ev_stop);
        
        char *temp = d_cells;
        d_cells = d_next_cells;
        d_next_cells = temp;

      } else if (method == OPENCL) {
        // No need to write data every frame - data is already on device
        // Just run the kernel with current buffers
        if (double_dim) {
          clSetKernelArg(grid_2d_kernel, 0, sizeof(cl_mem), &opencl_d_cells);
          clSetKernelArg(grid_2d_kernel, 1, sizeof(cl_mem),
                         &opencl_d_next_cells);
          clSetKernelArg(grid_2d_kernel, 2, sizeof(int), &GRID_X);
          clSetKernelArg(grid_2d_kernel, 3, sizeof(int), &GRID_Y);

          err_cl = clEnqueueNDRangeKernel(queue, grid_2d_kernel, 2, NULL,
                                          global_work_size_2d,
                                          local_work_size_2d, 0, NULL, NULL);
          if (err_cl != CL_SUCCESS) {
            std::cerr << "OpenCL 2D kernel launch failed, error: " << err_cl
                      << std::endl;
            return -1;
          }
        } else {
          clSetKernelArg(grid_1d_kernel, 0, sizeof(cl_mem), &opencl_d_cells);
          clSetKernelArg(grid_1d_kernel, 1, sizeof(cl_mem),
                         &opencl_d_next_cells);
          clSetKernelArg(grid_1d_kernel, 2, sizeof(int), &GRID_X);
          clSetKernelArg(grid_1d_kernel, 3, sizeof(int), &GRID_Y);

          err_cl = clEnqueueNDRangeKernel(queue, grid_1d_kernel, 1, NULL,
                                          global_work_size, local_work_size, 0,
                                          NULL, NULL);
          if (err_cl != CL_SUCCESS) {
            std::cerr << "OpenCL 1D kernel launch failed, error: " << err_cl
                      << std::endl;
            return -1;
          }
        }

        // Wait for kernel to complete
        clFinish(queue);

        // Swap the OpenCL buffers for next iteration
        cl_mem temp = opencl_d_cells;
        opencl_d_cells = opencl_d_next_cells;
        opencl_d_next_cells = temp;

        // Read the current state for display (from the buffer that now contains
        // results)
        clEnqueueReadBuffer(queue, opencl_d_cells, CL_TRUE, 0,
                            GRID_X * GRID_Y * sizeof(char), cells.data(), 0,
                            NULL, NULL);

        clFinish(queue);
      }

      draw_grid(GRID_X, GRID_Y);

      glfwSwapBuffers(window);
      glfwPollEvents();

      frame_end = std::chrono::high_resolution_clock::now();

      app_end = frame_end;
      cells_proccesed += GRID_X * GRID_Y;

      auto time_between_frames =
          std::chrono::duration_cast<std::chrono::microseconds>(frame_end -
                                                                frame_start);

      auto time_app = std::chrono::duration_cast<std::chrono::microseconds>(
          app_end - app_start);

      double cells_per_second =
          cells_proccesed / (time_app.count() / 1000000.f);

      int count = time_between_frames.count();

      if (count == 0) {
        count = 1000000;
      }

      int fps = 1000000 / count;

      std::ostringstream titleStream;
      titleStream << "GameOfLife - FPS: " << fps
                  << " Cells per second: " << cells_per_second;

      glfwSetWindowTitle(window, titleStream.str().c_str());
    }

    glfwDestroyWindow(window);
    glfwTerminate();
  } else {

    while (running) {

      if (method == CPU) {

        game_of_life_cpu();
        app_end = std::chrono::high_resolution_clock::now();

      } else if (method == CUDA) {
        cudaEventCreate(&ev_start);
        cudaEventCreate(&ev_stop);
        cudaEventRecord(ev_start, 0);
        if (double_dim) {
          game_of_life_kernel_2d<<<grid_size_2d, block_size_2d, 0, 0>>>(
              d_cells, d_next_cells, GRID_X, GRID_Y);
        } else {
          game_of_life_kernel<<<grid_size, block_size, 0, 0>>>(
              d_cells, d_next_cells, GRID_X, GRID_Y);
        }

        // cudaDeviceSynchronize();
        
        cudaMemcpy(cells.data(), d_next_cells, GRID_X * GRID_Y * sizeof(char),
        cudaMemcpyDeviceToHost);

        cudaEventRecord(ev_stop, 0);
        cudaEventSynchronize(ev_stop);
        
        err_cuda = cudaGetLastError();
        if (err_cuda != cudaSuccess) {
          std::cerr << "CUDA kernel launch failed: "
          << cudaGetErrorString(err_cuda) << std::endl;
          return -1;
        }
        
        app_end = std::chrono::high_resolution_clock::now();

        char *temp = d_cells;
        d_cells = d_next_cells;
        d_next_cells = temp;
      } else if (method == OPENCL) {

        if (double_dim) {
          clSetKernelArg(grid_2d_kernel, 0, sizeof(cl_mem), &opencl_d_cells);
          clSetKernelArg(grid_2d_kernel, 1, sizeof(cl_mem),
                         &opencl_d_next_cells);
          clSetKernelArg(grid_2d_kernel, 2, sizeof(int), &GRID_X);
          clSetKernelArg(grid_2d_kernel, 3, sizeof(int), &GRID_Y);

          err_cl = clEnqueueNDRangeKernel(queue, grid_2d_kernel, 2, NULL,
                                          global_work_size_2d,
                                          local_work_size_2d, 0, NULL, NULL);
          if (err_cl != CL_SUCCESS) {
            std::cerr << "OpenCL 2D kernel launch failed, error: " << err_cl
                      << std::endl;
            return -1;
          }
        } else {
          clSetKernelArg(grid_1d_kernel, 0, sizeof(cl_mem), &opencl_d_cells);
          clSetKernelArg(grid_1d_kernel, 1, sizeof(cl_mem),
                         &opencl_d_next_cells);
          clSetKernelArg(grid_1d_kernel, 2, sizeof(int), &GRID_X);
          clSetKernelArg(grid_1d_kernel, 3, sizeof(int), &GRID_Y);

          err_cl = clEnqueueNDRangeKernel(queue, grid_1d_kernel, 1, NULL,
                                          global_work_size, local_work_size, 0,
                                          NULL, NULL);
          if (err_cl != CL_SUCCESS) {
            std::cerr << "OpenCL 1D kernel launch failed, error: " << err_cl
                      << std::endl;
            return -1;
          }
        }

        clFinish(queue);
        
        // Read the current state for display (from the buffer that now contains
        // results)
        clEnqueueReadBuffer(queue, opencl_d_cells, CL_TRUE, 0,
          GRID_X * GRID_Y * sizeof(char), cells.data(), 0,
          NULL, NULL);
        clFinish(queue);
        
        // Swap the OpenCL buffers for next iteration
        cl_mem temp = opencl_d_cells;
        opencl_d_cells = opencl_d_next_cells;
        opencl_d_next_cells = temp;
        
        app_end = std::chrono::high_resolution_clock::now();
      }

      cells_proccesed += GRID_X * GRID_Y;

      auto time_app = std::chrono::duration_cast<std::chrono::microseconds>(
          app_end - app_start);

      double cells_per_second =
          cells_proccesed / (time_app.count() / 1000000.f);

      std::ostringstream titleStream;

      std::cout << "\rCells per second: "
                << static_cast<unsigned long long>(cells_per_second)
                << std::flush;
    }
  }

  std::cout << "\nMain loop ended" << std::endl;

  if (method == CUDA) {
    cudaFree(d_cells);
    cudaFree(d_next_cells);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    cudaDeviceReset();

    std::cout << "Freed CUDA resources" << std::endl;
  } else if (method == OPENCL) {
    clReleaseMemObject(opencl_d_cells);
    clReleaseMemObject(opencl_d_next_cells);

    clReleaseKernel(randomize_kernel);
    clReleaseProgram(randomize_program);

    clReleaseKernel(grid_1d_kernel);
    clReleaseProgram(grid_1d_program);

    clReleaseKernel(grid_2d_kernel);
    clReleaseProgram(grid_2d_program);

    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    std::cout << "Freed OPENCL resources" << std::endl;
  }

  return 0;
}
