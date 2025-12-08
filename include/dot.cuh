#pragma once

#include <vector>

__global__ void dot_kernel(float *out, const float *a, const float *b, size_t n);

__host__ float dot_host(const std::vector<float> &a, const std::vector<float> &b);