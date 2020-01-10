#include "render.hpp"
#include <spdlog/spdlog.h>
#include <cassert>

[[gnu::noinline]]
void _abortError(const char* msg, const char* fname, int line)
{
  cudaError_t err = cudaGetLastError();
  spdlog::error("{} ({}, line: {})", msg, fname, line);
  spdlog::error("Error {}: {}", cudaGetErrorName(err), cudaGetErrorString(err));
  std::exit(1);
}

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
        file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}
#define abortError(msg) _abortError(msg, __FUNCTION__, __LINE__)


struct rgba8_t {
  std::uint8_t r;
  std::uint8_t g;
  std::uint8_t b;
  std::uint8_t a;
};

__device__ vec3 color(const ray& r) {
    vec3 unit_direction = unit_vector(r.direction());
    float t = 0.5f*(unit_direction.y() + 1.0f);
    return (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
}



// Device code
__global__ void mykernel(char* buffer, int width, int height, size_t pitch, 
    vec3 lower_left_corner, vec3 horizontal, vec3 vertical, vec3 origin)
{
  float denum = width * width + height * height;

  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  uchar4*  lineptr = (uchar4*)(buffer + y * pitch);
  float u = float(x) / float(width);
  float v = float(y) / float(height);

  ray r(origin, lower_left_corner + u*horizontal + v*vertical);
  vec3 color = color(r);
  lineptr[x] = {color.r(), color.g(), color.b(), 255};
}

void render(char* hostBuffer, int width, int height, std::ptrdiff_t stride, int n_iterations)
{
  cudaError_t rc = cudaSuccess;

  // Allocate device memory
  char*  devBuffer;
  size_t pitch;

  checkCudaErrors(cudaMallocPitch(&devBuffer, &pitch, width *
        sizeof(rgba8_t), height));
  // Run the kernel with blocks of size 64 x 64
  {
    int bsize = 64;
    int w     = std::ceil((float)width / bsize);
    int h     = std::ceil((float)height / bsize);

    spdlog::debug("running kernel of size ({},{})", w, h);

    dim3 dimBlock(bsize, bsize);
    dim3 dimGrid(w, h);
    mykernel<<<dimGrid, dimBlock>>>(devBuffer, width, height, pitch);

    if (cudaPeekAtLastError())
      abortError("Computation Error");
  }

  // Copy back to main memory
  checkCudaErrors(cudaMemcpy2D(hostBuffer, stride, devBuffer, pitch, width
        * sizeof(rgba8_t), height, cudaMemcpyDeviceToHost));
  // Free
  checkCudaErrors(cudaFree(devBuffer));
}
