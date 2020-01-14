#pragma once

class object
{
public:
    object(unsigned nb_triangle, vec3 Kd, vec3 Ka, vec3 Ks)
      : nb_triangle(nb_triangle), Kd(Kd), Ka(Ka), Ks(Ks)
    {}
    object() {}

    unsigned nb_triangle;
    //float Ns;
    vec3 Kd;
    vec3 Ka;
    vec3 Ks;
};

class scene
{
public:
    scene(object* objs_m, unsigned nb_objs, vec3* vtx_m, unsigned nb_vtx)
      : objs_m(objs_m), nb_objs(nb_objs), vtx_m(vtx_m), nb_vtx(nb_vtx)
    {}

    object* objs_m;
    unsigned nb_objs;
    vec3* vtx_m;
    unsigned nb_vtx;
};
