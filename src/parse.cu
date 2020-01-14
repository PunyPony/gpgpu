#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <utility>
#include <cassert>


#include "render.hpp"
#include "parse.hpp"
#include "vec.hpp"
#include "triangle.hpp"
#include "hitable.hpp"
#include "hitable_list.hpp"
#include "material.hpp"


int isWord(char c)
{
  return (c != ' ') && (c != '\n') && (c != '\t');
}

hitable **parse_obj(std::string file_in)
{
  std::ifstream in(file_in, std::ios::in);
  if (!in)
  {
    std::cerr << "Cannot open " << file_in << std::endl;
    std::exit(1);
  }

 std::vector<vec3> v_list;
  std::vector<vec3> vn_list;
//  struct camera camera;
//  struct scene scene;

  std::string line;
  while (std::getline(in, line))
  {
    std::istringstream iss(line);
    std::string type;
    char *t;
    iss >> type;

    if (type == "v")
    {
      vec3 vec;
      std::sscanf(iss.str().c_str(), "%s %f %f %f", &t, &vec.e[0], &vec.e[1], &vec.e[2]);
      v_list.push_back(vec);
    }
    /*else if (type == "vn")
    {
      vec3 vec;
      std::sscanf(iss.str().c_str(), "%s %f %f %f", &t, &vec.x, &vec.y, &vec.z);
      vn_list.push_back(vec);
    }
    else if (type == "object")
    {
      obj = std::make_pair(v_list, vn_list);
      objects.push_back(obj);
      obj = obj_t();
    }
    else if (type == "camera")
    {
      camera_t c;
      std::sscanf(iss.str().c_str(), "%s %d %d %f", &t, &c.w, &c.h, &c.pos, &c.u, &c.w, &c.w);
      vn_list.push_back(vec);
    }*/

//      if (str == "camera")
//      {
//        int w;
//        int h;
//        vec3 pos;
//        vec3 u;
//        vec3 v;
//        float fov;
//      }
  }
//  obj = std::make_pair(v_list, vn_list);
  assert(v_list.size() % 3 == 0);
  unsigned nb_triangle = v_list.size() / 3;
  hitable **d_list;
  checkCudaErrors(cudaMalloc((void **)&d_list, nb_triangle*sizeof(hitable *)));
  for (unsigned i = 0; i < nb_triangle; i++) {
    d_list[i] = new triangle(v_list[3*i],
                             v_list[3*i+1],
                             v_list[3*i+2],
                             new lambertian(vec3(1.0, 0.0, 0.0)));
  }
  hitable **d_world;
  checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hitable *)));
  *d_world = new hitable_list(d_list, nb_triangle);
  return d_world;
}
