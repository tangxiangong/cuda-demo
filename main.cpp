#include <iostream>
#include <cuda_runtime.h>

#include "random.hpp"
#include "dot.h"

int main()
{
    size_t n = 1 << 20;
    
    auto a = randn<float>(n).value();
    auto b = randn<float>(n).value();

    float out = dot_cuda(a, b).value();

    std::cout << out << std::endl;

    return 0;
}