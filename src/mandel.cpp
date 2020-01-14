#include <cstddef>
#include <memory>

#include <png.h>

#include <CLI/CLI.hpp>
#include <spdlog/spdlog.h>
#include "render.hpp"


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


// Usage: ./mandel
int main(int argc, char** argv)
{
  if (argc != 2)
  {
    std::cout << "Usage : ./ray file.obj" << std::endl;
    return 0;
  }

  std::string filename = "output.png";
  unsigned width = 600;
  unsigned height = 300;
  unsigned ns = 50;
  unsigned bsize=32;

   // Create buffer
  //constexpr unsigned num_pixels = width * height
  constexpr int kRGBASize = 4;
  int stride = width * kRGBASize;
  auto buffer = std::make_unique<std::byte[]>(height * stride);

  // Rendering
  //spdlog::info("Runnging with (w={},h={},ns={}).", width, height, ns);
  render(argv[1], reinterpret_cast<char*>(buffer.get()), width, height, ns, stride, bsize);

  // Save
  write_png(buffer.get(), width, height, stride, filename.c_str());
  //spdlog::info("Output saved in {}.", "output.png");
}

