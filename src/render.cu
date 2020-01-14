#include <spdlog/spdlog.h>
#include <curand_kernel.h>
#include <iostream>
#include <cassert>
#include <stdio.h>
#include <time.h>
#include <float.h>

#include "hitable.hpp"
#include "render.hpp"
#include "vec.hpp"
#include "ray.hpp"
#include "hitable_list.hpp"
#include "camera.hpp"
#include "sphere.hpp"
#include "triangle.hpp"
#include "material.hpp"
#include "parse.hpp"

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
  if (result) {
    std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
      file << ":" << line << " '" << func << "' \n";
    // Make sure we call CUDA Device Reset before exiting
    cudaDeviceReset();
    exit(99);
  }
}


struct rgba8_t {
  std::uint8_t r;
  std::uint8_t g;
  std::uint8_t b;
  std::uint8_t a;
};

__global__ void load_world(object* objs, unsigned nb_objs, vec3* vtx, unsigned nb_vtx,
                           hitable **d_world, camera **d_camera, unsigned width, unsigned height) {
  unsigned i_vtx = 0;
  hitable **d_list = new hitable*[nb_vtx];
  for (unsigned i = 0; i < nb_objs; i++)
  {
    unsigned nb_triangle = objs[i].nb_triangle;
    for (unsigned t = 0; t < nb_triangle; t++)
    {
      d_list[i_vtx] = new triangle(vtx[i_vtx*3],
                                   vtx[i_vtx*3+1],
                                   vtx[i_vtx*3+2],
                                   new lambertian(objs[i].Ka));
      //vec3 ch = ((lambertian*)((triangle*)d_list[t])->mat_ptr)->albedo;
      //printf("Ka : %f %f %f\n", ch[0], ch[1], ch[2]);
      i_vtx += 1;
    }
    printf("Ka : %f %f %f\n", objs[i].Ka[0], objs[i].Ka[1], objs[i].Ka[2]);
  }
  d_world[0] = new hitable_list(d_list, nb_vtx/3);

        vec3 lookfrom(0, 0, -5);
        vec3 lookat(0,0,0);
        float dist_to_focus = 10.0; (lookfrom-lookat).length();
        float aperture = 0.;
        *d_camera = new camera(lookfrom,
                                 lookat,
                                 vec3(0,1,0),
                                 70.0,
                                 float(width)/float(height),
                                 aperture,
                                 dist_to_focus);
}

__global__ void create_world(unsigned nb_objs, hitable **d_world, camera
    **d_camera, unsigned width, unsigned height) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        hitable **d_list = new hitable*[8];
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

        d_list[5] = new triangle(vec3(2,0,0.75), vec3(2,1,0.75),
            vec3(0.5,0.0,-20), new metal(vec3(1.0, 0.0, 0.0), 0.2));


        d_list[6] = new triangle(vec3(-0.5,-0.25,0.75), vec3(0,0.33,0.75),
              vec3(0.5,-0.25,0.75), new dielectric(1.3));

        *d_world = new hitable_list(d_list,7);
        
        vec3 lookfrom(0, 0, 2);
        vec3 lookat(0,0,0);
        float dist_to_focus = 10.0; //(lookfrom-lookat).length();
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

__global__ void free_world(unsigned nb_objs, hitable **d_world, camera **d_camera) {
    for (unsigned obj = 0; obj < 1; obj++)
    {
      hitable_list *d_obj = (hitable_list*)d_world[obj];
      hitable **d_list = d_obj->list;
      for(int i = 0; i < d_obj->list_size; i++) {
        //delete ((triangle *)d_list[i])->mat_ptr;
        delete d_list[i]->get_mat_ptr();
        delete d_list[i];
      }
      delete d_obj->list;
      delete d_world[obj];
    }


    delete *d_camera;
}


//max depth 50
__device__ vec3 color(const ray& r, hitable **world, curandState *local_rand_state) {
  ray cur_ray = r;
  vec3 cur_attenuation = vec3(1.0,1.0,1.0);
  for(int i = 0; i < 50; i++) {
    hit_record rec;
    if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) 
    {
      ray scattered;
      vec3 attenuation;
      if(rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
        cur_attenuation *= attenuation;
        cur_ray = scattered;
      }
      else
        return vec3(0.0,0.0,0.0);
    }
    else
      return cur_attenuation;
  }
  return vec3(0.0,0.0,0.0); // exceeded recursion, bounces 50 times
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

  /*extern __shared__  hitable* shared_world[];
  for (unsigned i=0; i<7;i++)
    shared_world[i] = world[i];
  __syncthreads();

  shared_world[0];*/

  uchar4*  lineptr = (uchar4*)(buffer + (height-y) * pitch); //reversed 

  curandState local_rand_state = rand_state[y*width + x];
  vec3 col(0,0,0);
  // anti aliasing
  for(int s=0; s < ns; s++) {
    float u = float(x + curand_uniform(&local_rand_state)) / float(width);
    float v = float(y + curand_uniform(&local_rand_state)) / float(height);
    ray r = (*cam)->get_ray(u,v, &local_rand_state);
    col += color(r,world, &local_rand_state);
  }

  col = col/float(ns);
  lineptr[x] = {col.r_sqrt(), col.g_sqrt(), col.b_sqrt(), 255};
}

void render(char* in_filename, char* hostBuffer, unsigned width, unsigned height, unsigned ns,
    std::ptrdiff_t stride, unsigned bsize)
{
  //cudaError_t rc = cudaSuccess;

  // Allocate device memory
  char*  devBuffer;
  size_t pitch;

  // allocate random state
  curandState *d_rand_state;
  checkCudaErrors(cudaMalloc((void **)&d_rand_state, width*height*sizeof(curandState)));

  // make our world of hitables & the camera
  unsigned nb_objs = 1;
  unsigned nb_vtx = 0;
  hitable **d_world;
  checkCudaErrors(cudaMalloc((void **)&d_world, nb_objs*sizeof(hitable *)));
  camera **d_camera;
  checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));

  if (std::string(in_filename) == "")
  {
    create_world<<<1,1>>>(nb_objs,d_world,d_camera, width, height);
  }
  else
  {
  scene scene1 = parse_obj(in_filename);
  nb_objs = scene1.nb_objs;
  nb_vtx = scene1.nb_vtx;

  object *objs_cuda;
  checkCudaErrors(cudaMalloc((void **)&objs_cuda, nb_objs*sizeof(object)));
  checkCudaErrors(cudaMemcpy(objs_cuda, scene1.objs_m, nb_objs*sizeof(object), cudaMemcpyHostToDevice));
  vec3 *vtx_cuda;
  checkCudaErrors(cudaMalloc((void **)&vtx_cuda, nb_vtx*sizeof(vec3)));
  checkCudaErrors(cudaMemcpy(vtx_cuda, scene1.vtx_m, nb_vtx*sizeof(vec3), cudaMemcpyHostToDevice));
  delete scene1.objs_m;
  delete scene1.vtx_m;
  load_world<<<1,1>>>(objs_cuda, nb_objs, vtx_cuda, nb_vtx,
                      d_world, d_camera, width, height);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaFree(objs_cuda));
  checkCudaErrors(cudaFree(vtx_cuda));
  }

  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
  
  checkCudaErrors(cudaMallocPitch(&devBuffer, &pitch, width *
        sizeof(rgba8_t), height));

  checkCudaErrors(cudaGetLastError());

  // Run the kernel with blocks of size bsize*bsize
  {
    //int bsize = 32;
    int w     = std::ceil((float)width / bsize);
    int h     = std::ceil((float)height / bsize);

    spdlog::debug("running kernel of size ({},{})", w, h);

    dim3 dimBlock(bsize, bsize);
    dim3 dimGrid(w, h);

    //random init
    render_init<<<dimBlock, dimGrid>>>(width, height, d_rand_state);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    //raytracing;
    mykernel<<<dimGrid, dimBlock, 1*sizeof(hitable *)>>>(devBuffer, width, height, pitch, 
        ns, d_camera, d_world, d_rand_state);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
  }

  // Copy back to main memory
  checkCudaErrors(cudaMemcpy2D(hostBuffer, stride, devBuffer, pitch, width
        * sizeof(rgba8_t), height, cudaMemcpyDeviceToHost));
  // Free
  checkCudaErrors(cudaDeviceSynchronize()); // ?

  free_world<<<1,1>>>(nb_objs,d_world,d_camera);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaFree(devBuffer));
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaFree(d_camera));
  checkCudaErrors(cudaFree(d_world));
  checkCudaErrors(cudaFree(d_rand_state));

  // useful for cuda-memcheck --leak-check full
  cudaDeviceReset();
}
