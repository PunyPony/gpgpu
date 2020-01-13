#pragma once
#include <cstddef>
#include <memory>

void render(char* buffer, unsigned width, unsigned height, 
    unsigned ns, std::ptrdiff_t stride, unsigned bsize);
