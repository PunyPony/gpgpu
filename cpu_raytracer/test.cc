#include <iostream>
#include "float.h"
#include"main.hh"


#include <cstddef>
#include <memory>






int main() {
    int nx = 120;
    int ny = 800;
    int ns = 10;
    auto buffer = std::make_unique< std::byte[]>(ny * nx*4);
//    render(buffer,ny,nx,ns);  
    test(); 
}
