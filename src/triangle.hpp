#pragma once
#include "hitable.hpp"

class triangle: public hitable  {
    public:
        __device__ triangle() {}
        __device__ triangle(vec3 v0, vec3 v1,vec3 v2, material *m) : v0(v0), v1(v1),v2(v2), mat_ptr(m)  {};
        __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
        vec3 v0;
        vec3 v1;
        vec3 v2;

        material *mat_ptr;
};

__device__ bool triangle::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {

    vec3 v0v1  = v1 - v0;   // edge 0
    vec3 v0v2  = v2 - v0;  //edge 1 
    vec3 N = cross(v0v1,v0v2); // this is the triangle normal 
    //normalize(C);
    float d = dot(N,v0);  // get the distance
    auto dir = r.direction();
    auto orig = r.origin();

    float t = (dot(N,orig) + d) / dot(N,dir);

    float a = dot(r.direction(), r.direction());
    t = abs(t / a);
    vec3 P = orig +t *dir;
    // egde 0
    vec3 edge0 = v1 - v0;
    vec3 vp0 = P - v0;
    auto C = cross(edge0,vp0);
    float nd = dot(N,dir);
    if(fabs(nd) < 0.0001) return false;
    
    if (dot(N,C) < 0  ) return false;
    // egde 1
    vec3 edge1 = v2 - v1;
    vec3 vp1 = P - v1;
    C = cross(edge1,vp1);
    if (dot(N,C) < 0  ) return false;
    // egde 2
    vec3 edge2 = v0 - v2;
    vec3 vp2 = P - v2;

    C = cross(edge2,vp2);

    if (dot(N,C) < 0  ) return false;


    if (t < t_max && t > t_min) {

      rec.t = t;
      rec.p = r.point_at_parameter(rec.t);
      rec.normal = N ;
      rec.mat_ptr = mat_ptr;
      return true;
    }
    return false;  
}

