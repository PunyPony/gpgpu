#include "render.hpp"
#include <vector>
#include <benchmark/benchmark.h>
#include <iostream>
unsigned width = 1200;
unsigned height = 600;
unsigned ns = 10;
constexpr int kRGBASize = 4;
int stride = width * kRGBASize;
auto buffer = std::make_unique<std::byte[]>(height * stride);


static void BM_Rendering_gpu(benchmark::State& st)
{

  // Rendering
  for(auto _:st)
    render("", reinterpret_cast<char*>(buffer.get()), width, height, ns, stride, 32);
  st.counters["frame_rate"] = benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
}

static void BM_Rendering_cpu(benchmark::State& st)
{
  for(auto _:st)
    render_cpu(buffer, height, width, ns);
  st.counters["frame_rat"] = benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
}


BENCHMARK(BM_Rendering_gpu)
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();

BENCHMARK(BM_Rendering_cpu)
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();



BENCHMARK_MAIN();
