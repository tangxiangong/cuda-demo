#include "dot.cuh"
#include <cuda_runtime.h>

__global__ void dot_kernel(float *out, const float *a, const float *b,
                           size_t n)
{
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;

    float val = 0.0f;
    if (idx < n)
    {
        val = a[idx] * b[idx];
    }

    __shared__ float cache[256];
    cache[threadIdx.x] = val;
    __syncthreads();

    for (size_t s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            cache[threadIdx.x] += cache[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        atomicAdd(out, cache[0]);
    }
}

__host__ float dot_host(const std::vector<float> &a,
                        const std::vector<float> &b)
{
    size_t n = a.size();

    float *a_device;
    float *b_device;
    float *out_device;

    cudaMalloc(&a_device, n * sizeof(float));
    cudaMalloc(&b_device, n * sizeof(float));
    cudaMalloc(&out_device, sizeof(float));

    cudaMemcpy(a_device, a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_device, b.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(out_device, 0, sizeof(float));

    size_t block_size = 256;
    size_t block_num = (n + block_size - 1) / block_size;

    dot_kernel<<<block_num, block_size>>>(out_device, a_device, b_device, n);

    float out;
    cudaMemcpy(&out, out_device, sizeof(float), cudaMemcpyDeviceToHost);
    return out;
}