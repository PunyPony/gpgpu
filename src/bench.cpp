#include "render.hpp"
#include <vector>
#include <benchmark/benchmark.h>
#include <iostream>

unsigned width = 1200;
unsigned height = 600;
unsigned ns = 100;
constexpr int kRGBASize = 4;
int stride = width * kRGBASize;
auto buffer = std::make_unique<std::byte[]>(height * stride);

static void BM_Rendering_gpu(benchmark::State& st)
{
  unsigned p = 0;
  // Rendering
  for(auto _:st)
  {
    p++;
    render(reinterpret_cast<char*>(buffer.get()), width, height, ns, stride, 32);
  }
  //st.counters["frame_rate"] = benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
  std::cout << p << std::endl;
  st.counters["p"] = p;
  st.counters["frame_rate"] = benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
  st.counters["test"] = benchmark::Counter(st.real_time(), benchmark::Counter::kAvgThreads);
}

BENCHMARK(BM_Rendering_gpu)
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();
BENCHMARK_MAIN();
