#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <utility>
#include <cassert>
#include <stdlib.h>

#include "hitable.hpp"
#include "object.hpp"

scene parse_obj(std::string file_in)
{
  std::ifstream in(file_in, std::ios::in);
  if (!in)
  {
    std::cerr << "Cannot open " << file_in << std::endl;
    std::exit(1);
  }

  std::vector<vec3> v_list;
  std::vector<vec3> vn_list;
  std::vector<object> objs;
//  struct camera camera;
//  struct scene scene;

  unsigned nb_triangle = 0;
  vec3 Kd(0, 0, 0);
  vec3 Ka(0, 0, 0);
  vec3 Ks(0, 0, 0);
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
    else if (type == "object")
      std::sscanf(iss.str().c_str(), "%s %d", &t, &nb_triangle);
    else if (type == "Kd")
      std::sscanf(iss.str().c_str(), "%s %f %f %f", &t, &Kd.e[0], &Kd.e[1], &Kd.e[2]);
    else if (type == "Ka")
      std::sscanf(iss.str().c_str(), "%s %f %f %f", &t, &Ka.e[0], &Ka.e[1], &Ka.e[2]);
    else if (type == "Ks")
    {
      std::sscanf(iss.str().c_str(), "%s %f %f %f", &t, &Ks.e[0], &Ks.e[1], &Ks.e[2]);
      objs.push_back(object(nb_triangle, Kd, Ka, Ks));
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



  /*assert(v_list.size() % 3 == 0);
  unsigned nb_triangle = v_list.size() / 3;
  std::cout << "nb_triangle : " << nb_triangle*sizeof(hitable *) << std::endl;
  hitable **d_list = (hitable **)malloc(nb_triangle*sizeof(hitable *));
  std::cout << "NICE MALLOC" << std::endl;
  for (unsigned i = 0; i < nb_triangle; i++) {
    std::cout << "index : " << i << std::endl;
    std::cout << "val : " << *d_list << std::endl;
    d_list[i] = new triangle(v_list[3*i],
                             v_list[3*i+1],
                             v_list[3*i+2],
                             new lambertian(vec3(1.0, 0.0, 0.0)));
  }

  hitable **d_world = (hitable **)malloc(sizeof(hitable *));
  *d_world = new hitable_list(d_list, nb_triangle);
  return d_world;*/

  assert(v_list.size() % 3 == 0);
  unsigned nb_vtx = v_list.size();
  vec3* vtx_m = new vec3[nb_vtx];
  for (unsigned i = 0; i < nb_vtx; i++) {
    std::cout << "index : " << i << std::endl;
    std::cout << "val : " << v_list[i] << std::endl;
    vtx_m[i] = v_list[i];
  }

  unsigned nb_objs = objs.size();
  object* objs_m = new object[nb_objs];
  for (unsigned i = 0; i < nb_objs; i++)
  {
    std::cout << "val : " << objs[i].Ka[0] << std::endl;
    objs_m[i] = objs[i];
  }

  return scene(objs_m, nb_objs, vtx_m, nb_vtx);
}
