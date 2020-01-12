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
#include "camera.hpp"
#include "hitable_list.hpp"
#include <curand_kernel.h>
#include "material.hpp"
# include "triangle.hpp"
/*
   [[gnu::noinline]]
   void _abortError(const char* msg, const char* fname, int line)
   {
   cudaError_t err = cudaGetLastError();
   spdlog::error("{} ({}, line: {})", msg, fname, line);
   spdlog::error("Error {}: {}", cudaGetErrorName(err), cudaGetErrorString(err));
   std::exit(1);
   }
 */


#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
#define abortError(msg) _abortError(msg, __FUNCTION__, __LINE__)
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
  if (result) {
    std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
      file << ":" << line << " '" << func << "' \n";
    // Make sure we call CUDA Device Reset before exiting
    cudaDeviceReset();
    exit(99);
  }
}


/*
#define RANDVEC3 vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))
__device__ vec3 random_in_unit_sphere(curandState *local_rand_state) {
  vec3 p;
  do {
    p = 2.0f*RANDVEC3 - vec3(1,1,1);
  } while (p.squared_length() >= 1.0f);
  return p;
}*/


struct rgba8_t {
  std::uint8_t r;
  std::uint8_t g;
  std::uint8_t b;
  std::uint8_t a;
};

__global__ void create_world(hitable **d_list, hitable **d_world, camera
    **d_camera, unsigned width, unsigned height) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_list[0] = new sphere(vec3(0,0,0), 0.5,
                                new lambertian(vec3(0.1, 0.2, 0.5)));
        d_list[1] = new sphere(vec3(0,-100.5,0), 100,
                                new lambertian(vec3(0.78, 0.18, 0.4)));

        d_list[2] = new sphere(vec3(1,1,0), 0.5,
                               new lambertian(vec3(0.1, 0.8, 0.4)));


        d_list[3] = new sphere(vec3(1,0,0), 0.5,
                                new metal(vec3(0.8, 0.6, 0.2), 0.2));
        d_list[4] = new sphere(vec3(0,1,0), 0.5,
                                 new dielectric(1.5));

        d_list[5] = new triangle(vec3(0,0,1), vec3(0,1,1), vec3(1,0,1),
                               new lambertian(vec3(1.0, 0.0, 0.0)));

/*
        d_list[5] = new sphere(vec3(0,0,1), 0.25,
                               new lambertian(vec3(1.0, 0.0, 0.0)));
*/


        d_list[6] = new triangle(vec3(3,0,1), vec3(0,1,1), vec3(1,0,1),
                                new metal(vec3(0.8, 0.6, 0.2), 0.2));
        d_list[7] = new triangle(vec3(0,10,0), vec3(10,0,0), vec3(10,10,0),
                                 new dielectric(1.5));


        *d_world = new hitable_list(d_list,8);
        
        vec3 lookfrom(0, 0, 2);
        vec3 lookat(0,0,0);
        float dist_to_focus = 10.0; (lookfrom-lookat).length();
        float aperture = 0.;
        *d_camera = new camera(lookfrom,
                                 lookat,
                                 vec3(0,1,0),
                                 100.0,
                                 float(width)/float(height),
                                 aperture,
                                 dist_to_focus);
    }
}

__global__ void free_world(hitable **d_list, hitable **d_world, camera **d_camera) {
    for(int i=0; i < 5; i++) {
        delete ((sphere *)d_list[i])->mat_ptr;
        delete d_list[i];
    }
    for(int i=5; i < 8; i++) {
        delete ((triangle *)d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete *d_world;
    delete *d_camera;
}


/*
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
   }*/

//max depth 50
__device__ vec3 color(const ray& r, hitable **world, curandState *local_rand_state) {
  ray cur_ray = r;
  vec3 cur_attenuation = vec3(1.0,1.0,1.0);
  for(int i = 0; i < 50; i++) {
    hit_record rec;
    if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
      ray scattered;
      vec3 attenuation;
      if(rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
        cur_attenuation *= attenuation;
        cur_ray = scattered;

      }
      else {
        return vec3(0.0,0.0,0.0);
      }
    }
    else {
      vec3 unit_direction = unit_vector(cur_ray.direction());
      float t = 0.5f*(unit_direction.y() + 1.0f);
      vec3 c = (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
      return cur_attenuation * c;
    }
  }
  return vec3(0.0,0.0,0.0); // exceeded recursion
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if((i >= max_x) || (j >= max_y)) return;
  int pixel_index = j*max_x + i;
  //Each thread gets same seed, a different sequence number, no offset
  curand_init(0, pixel_index, 0, &rand_state[pixel_index]);
}


// Device code
__global__ void mykernel(char* buffer, int width, int height, size_t pitch, 
    int ns, camera **cam, hitable **world, curandState *rand_state)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  uchar4*  lineptr = (uchar4*)(buffer + (height-y) * pitch);

  curandState local_rand_state = rand_state[y*width + x];
  vec3 col(0,0,0);
  for(int s=0; s < ns; s++) {
    float u = float(x + curand_uniform(&local_rand_state)) / float(width);
    float v = float(y + curand_uniform(&local_rand_state)) / float(height);
    ray r = (*cam)->get_ray(u,v, &local_rand_state);
    col += color(r, world, &local_rand_state);
  }
  //const ray& r, hitable **world, curandState *local_rand_state

  col = col/float(ns);
  lineptr[x] = {col.r_sqrt(), col.g_sqrt(), col.b_sqrt(), 255};
}

void render(char* hostBuffer, unsigned width, unsigned height, unsigned ns, std::ptrdiff_t stride)
{
  //cudaError_t rc = cudaSuccess;

  // Allocate device memory
  char*  devBuffer;
  size_t pitch;

  // allocate random state
  curandState *d_rand_state;
  checkCudaErrors(cudaMalloc((void **)&d_rand_state, width*height*sizeof(curandState)));

  // make our world of hitables & the camera
  hitable **d_list;
  checkCudaErrors(cudaMalloc((void **)&d_list, 8*sizeof(hitable *)));
  hitable **d_world;
  checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hitable *)));
  camera **d_camera;
  checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));
  create_world<<<1,1>>>(d_list,d_world,d_camera, width, height);

  checkCudaErrors(cudaMallocPitch(&devBuffer, &pitch, width *
        sizeof(rgba8_t), height));

  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  printf("allocate\n");
  // Run the kernel with blocks of size bsize*bsize
  {
    int bsize = 32;
    int w     = std::ceil((float)width / bsize);
    int h     = std::ceil((float)height / bsize);

    spdlog::debug("running kernel of size ({},{})", w, h);

    dim3 dimBlock(bsize, bsize);
    dim3 dimGrid(w, h);

    printf("random init\n");
    render_init<<<dimBlock, dimGrid>>>(width, height, d_rand_state);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    printf("raytracing\n");
    mykernel<<<dimGrid, dimBlock>>>(devBuffer, width, height, pitch, 
        ns, d_camera, d_world, d_rand_state);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
  }

  // Copy back to main memory
  checkCudaErrors(cudaMemcpy2D(hostBuffer, stride, devBuffer, pitch, width
        * sizeof(rgba8_t), height, cudaMemcpyDeviceToHost));
  // Free
  checkCudaErrors(cudaDeviceSynchronize()); // ?

  free_world<<<1,1>>>(d_list,d_world,d_camera);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaFree(devBuffer));
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaFree(d_camera));
  checkCudaErrors(cudaFree(d_world));
  checkCudaErrors(cudaFree(d_list));
  checkCudaErrors(cudaFree(d_rand_state));

  // useful for cuda-memcheck --leak-check full
  cudaDeviceReset();
}
