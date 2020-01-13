#include "render.hpp"
#include <vector>
#include <benchmark/benchmark.h>

unsigned width = 1200;
unsigned height = 600;
unsigned ns = 100;
constexpr int kRGBASize = 4;
int stride = width * kRGBASize;
auto buffer = std::make_unique<std::byte[]>(height * stride);

void BM_Rendering_gpu(benchmark::State& st)
{
  // Rendering
  for(auto _:st)
    render(reinterpret_cast<char*>(buffer.get()), width, height, ns, stride, 32);
  st.counters["frame_rate"] = benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
}

BENCHMARK(BM_Rendering_gpu)
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();
BENCHMARK_MAIN();
