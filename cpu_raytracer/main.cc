#include <iostream>
#include "sphere.h"
#include "hitable_list.h"
#include "float.h"
#include "camera.h"
#include "material.h"

#define MAXFLOAT FLT_MAX
#include "triangle.hpp"

#include <cstddef>
#include <memory>

#include <png.h>

//#include <CLI/CLI.hpp>
//#include <spdlog/spdlog.h>






vec3 color(const ray& r, hitable *world, int depth) {
    hit_record rec;
    if (world->hit(r, 0.001, MAXFLOAT, rec)) {
        ray scattered;
        vec3 attenuation;
        if (depth < 50 && rec.mat_ptr->scatter(r, rec, attenuation, scattered)) {
             return attenuation*color(scattered, world, depth+1);
        }
        else {
            return vec3(0,0,0);
        }
    }
    else {
        vec3 unit_direction = unit_vector(r.direction());
        float t = 0.5*(unit_direction.y() + 1.0);
        return (1.0-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
    }
}

void create_world(hitable **d_list, hitable **d_world, camera
    **d_camera, unsigned width, unsigned height) {

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



void write_png(const std::byte* buffer,
               int width,
               int height,
               int stride,
               const char* filename)
{
  png_structp png_ptr =
    png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);

  if (!png_ptr)
    return;

  png_infop info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr)
  {
    png_destroy_write_struct(&png_ptr, nullptr);
    return;
  }

  FILE* fp = fopen(filename, "wb");
  png_init_io(png_ptr, fp);

  png_set_IHDR(png_ptr, info_ptr,
               width,
               height,
               8,
               PNG_COLOR_TYPE_RGB_ALPHA,
               PNG_INTERLACE_NONE,
               PNG_COMPRESSION_TYPE_DEFAULT,
               PNG_FILTER_TYPE_DEFAULT);

  png_write_info(png_ptr, info_ptr);
  for (int i = 0; i < height; ++i)
  {
    png_write_row(png_ptr, reinterpret_cast<png_const_bytep>(buffer));
    buffer += stride;
  }

  png_write_end(png_ptr, info_ptr);
  png_destroy_write_struct(&png_ptr, nullptr);
  fclose(fp);
}



void  render(std::byte[]  buffer ) {
    int nx = 1200;
    int ny = 800;
    int ns = 10;


    hitable **list = new hitable *[7];

    hitable **world  = new hitable*[2];
    camera **cam = new camera*[2]; 
    create_world(list,world,cam,nx,ny);
    float R = cos(M_PI/4);
   int stride = nx * 4;
   for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            vec3 col(0, 0, 0);
            for (int s=0; s < ns; s++) {
                float u = float(i + drand48()) / float(nx);
                float v = float(j + drand48()) / float(ny);
                ray r = (*cam)->get_ray(u, v);
                vec3 p = r.point_at_parameter(2.0);
                col += color(r, *world,0);
            }
            col /= float(ns);
            col = vec3( sqrt(col[0]), sqrt(col[1]), sqrt(col[2]) );
            auto ir = int(255.99*col[0]);
            auto ig = int(255.99*col[1]);
            auto ib = int(255.99*col[2]);
            buffer[j*nx+ i*4+0] = std::byte{ir};
            buffer[j*nx+ i*4 + 1] = std::byte{ig};
            buffer[j*nx+ i*4 + 2] = std::byte{ib};
            buffer[j*nx+ i*4 +3] = std::byte{255};
       }
    }
  return buffer
}




int main() {

  auto buffer = std::make_unique< std::byte[]>(ny * stride);
  render(buffer);   
}
