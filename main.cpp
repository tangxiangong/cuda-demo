#include <iostream>
#include <cuda_runtime.h>

#include "random.hpp"
#include "dot.h"

int main()
{
    size_t n = 1 << 20;

    auto a = randn<float>(n).value();
    auto b = randn<float>(n).value();

    float out_cuda = dot_cuda(a, b).value();
    float out_cpu = dot_cpu(a, b).value();
    std::cout << out_cuda << std::endl;
    std::cout << out_cpu << std::endl;

    return 0;
}