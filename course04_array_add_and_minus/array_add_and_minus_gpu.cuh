#pragma once

__global__ void array_add_and_minus_gpu(const double* a, const double* b, const double* c, double* d) {
    unsigned long i = blockDim.x * blockIdx.x + threadIdx.x;
    // 当ARRAY_LEN不是blockDim.x整数倍时，通过if语句避免设备内存越界访问
    if (i < ARRAY_LEN) {
        d[i] = a[i] + b[i] - c[i];
    }
}