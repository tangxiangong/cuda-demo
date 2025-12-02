extern "C" __global__ void matmul_kernel(float *out, const float *a,
                                         const float *b, int N) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < N && col < N) {
    float sum = 0.0f;
    for (int k = 0; k < N; ++k) {
      sum += a[row * N + k] * b[k * N + col];
    }
    out[row * N + col] = sum;
  }
}
