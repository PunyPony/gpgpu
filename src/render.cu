#include "render.hpp"
#include <spdlog/spdlog.h>
#include <cassert>
#include <iostream>
#include <stdio.h>
#include <time.h>
#include <float.h>
#include "vec.hpp"
#include "ray.hpp"
#include "sphere.hpp"
#include "hitable_list.hpp"


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

__global__ void create_world(hitable **d_list, hitable **d_world) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *(d_list)   = new sphere(vec3(0,0,-1), 0.5);
        *(d_list+1) = new sphere(vec3(0,-100.5,-1), 100);
        *d_world    = new hitable_list(d_list,2);
    }
}

__global__ void free_world(hitable **d_list, hitable **d_world) {
    delete *(d_list);
    delete *(d_list+1);
    delete *d_world;
}

__device__ vec3 color(const ray& r, hitable **world) {
    hit_record rec;
    if ((*world)->hit(r, 0.0, FLT_MAX, rec)) {
        return 0.5f*vec3(rec.normal.x()+1.0f, rec.normal.y()+1.0f, rec.normal.z()+1.0f);
    }
    else {
        vec3 unit_direction = unit_vector(r.direction());
        float t = 0.5f*(unit_direction.y() + 1.0f);
        return (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
    }
}

// Device code
__global__ void mykernel(char* buffer, int width, int height, size_t pitch, 
    vec3 lower_left_corner, 
    vec3 horizontal, 
    vec3 vertical, 
    vec3 origin, 
    hitable **world)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  float denum = width * width + height * height;
  uchar4*  lineptr = (uchar4*)(buffer + y * pitch);
  
  float u = float(x) / float(width);
  float v = float(y) / float(height);

  ray r = ray(origin, lower_left_corner + u*horizontal + v*vertical);
  vec3 c = color(r, world);
  //printf("%f", c.r());
  lineptr[x] = {c.r(), c.g(), c.b(), 255};
}

void render(char* hostBuffer, int width, int height, std::ptrdiff_t stride)
{
  //cudaError_t rc = cudaSuccess;

  // Allocate device memory
  char*  devBuffer;
  size_t pitch;
 
  hitable **d_list;
  checkCudaErrors(cudaMalloc((void **)&d_list, 2*sizeof(hitable *)));
  hitable **d_world;
  checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hitable *)));
  create_world<<<1,1>>>(d_list,d_world);
  
   checkCudaErrors(cudaMallocPitch(&devBuffer, &pitch, width *
        sizeof(rgba8_t), height));
  
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());



  // Run the kernel with blocks of size 64 x 64
  {
    int bsize = 32;
    int w     = std::ceil((float)width / bsize);
    int h     = std::ceil((float)height / bsize);
    
    spdlog::debug("running kernel of size ({},{})", w, h);

    dim3 dimBlock(bsize, bsize);
    dim3 dimGrid(w, h);

    mykernel<<<dimGrid, dimBlock>>>(devBuffer, width, height, pitch,
    vec3(-2.0, -1.0, -1.0),
    vec3(4.0, 0.0, 0.0),
    vec3(0.0, 2.0, 0.0),
    vec3(0.0, 0.0, 0.0),
    d_world);

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
  }

  // Copy back to main memory
  checkCudaErrors(cudaMemcpy2D(hostBuffer, stride, devBuffer, pitch, width
        * sizeof(rgba8_t), height, cudaMemcpyDeviceToHost));
  // Free
  checkCudaErrors(cudaDeviceSynchronize());
  free_world<<<1,1>>>(d_list,d_world);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaFree(d_list));
  checkCudaErrors(cudaFree(d_world));
  checkCudaErrors(cudaFree(devBuffer));
  cudaDeviceReset();
}
