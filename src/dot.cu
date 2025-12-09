#include "dot.cuh"
#include <cuda_runtime.h>

__inline__ __device__ float warp_reduce_sum(float val)
{
    int mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}

__global__ void dot_kernel(float *out, const float *a, const float *b,
                           size_t n)
{
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;

    float val = 0.0f;
    if (idx < n)
    {
        val = a[idx] * b[idx];
    }

    val = warp_reduce_sum(val);

    __shared__ float warp_sums[32];

    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    if (lane == 0)
    {
        warp_sums[warp_id] = val;
    }

    __syncthreads();

    float block_sum = 0.0f;
    if (warp_id == 0)
    {
        if (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize)
        {
            block_sum = warp_sums[lane];
        }
        block_sum = warp_reduce_sum(block_sum);
    }

    if (threadIdx.x == 0)
    {
        atomicAdd(out, block_sum);
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