#pragma once
#include <cstddef>
#include <memory>
#include <iostream>

#ifdef __NVCC__

#define abortError(msg) _abortError(msg, __FUNCTION__, __LINE__)
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line);

#else

void render(char* buffer, unsigned width, unsigned height, 
    unsigned ns, std::ptrdiff_t stride, unsigned bsize);

void render_cpu(std::unique_ptr<std::byte[]>&  buffer,int ny, int nx,int ns );

#endif
