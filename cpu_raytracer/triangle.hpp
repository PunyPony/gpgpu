#pragma once
#include "hitable.h"

class triangle: public hitable  {
    public:
        triangle() {}
        triangle(vec3 v0, vec3 v1,vec3 v2, material *m) : v0(v0), v1(v1),v2(v2), mat_ptr(m)  {};
        virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
        vec3 v0;
        vec3 v1;
        vec3 v2;

        material *mat_ptr;
};

bool triangle::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {

    vec3 edge1  = v1 - v0;   // edge 0
    vec3 edge2  = v2 - v0;  //edge 1 
    vec3 h = cross(r.direction(),edge2);
    float epsilon = 0.001;
    float a = dot(edge1,h);
    if ( fabs(a) < epsilon) return false;
    float f = 1.0/a;
    vec3 s = r.origin() - v0;
    float u = f * dot(s,h);
    if ( u < 0.0 || u > 1.0) return false;
    vec3 q = cross(s,edge1);
    float v = f * dot(r.direction(),q);
    if ( v < 0.0 || u+v > 1.0) return false;
    float t = f * dot(edge2,q);
    if (t > epsilon && t <1 - epsilon)
    {
      if (t < t_max && t > t_min) {
        rec.t = t;
        rec.p = r.point_at_parameter(rec.t);
        vec3 normal_vector = cross(edge1,edge2);
        // cross product is oriented
        // normat at point p
        if ((dot(rec.p, normal_vector) / (rec.p.length()*normal_vector.length()))>0)
          rec.normal = rec.p - normal_vector;
        else
          rec.normal = rec.p + normal_vector;
        rec.mat_ptr = mat_ptr;
      return true;
      }
    }
    return false;
}

