#include <cstdio>

// CUDA核函数（CPU主机端调用，GPU设备端执行）
// CUDA核函数标识符：__global__
__global__ void hello_cuda_from_gpu() {
    printf("GPU: 你好, CUDA! (C++版)\n");
}

// 普通函数5（CPU主机端调用和执行）
void hello_cuda_from_cpu() {
    printf("CPU: 你好, CUDA! (C++版)\n");
}

int main() {
    // GPU: 你好, CUDA! (C++版)
    {
        const int GRID_DIM = 2;          // Grid(网格)大小（即1个Grid中的线程块数量）
        const int BLOCK_DIM = 8;         // Block(线程块)大小（即1个Block中的线程数量）
        // CUDA核函数调用, 核函数配置参数<<<...>>>, 核函数总线程数为2*8=16
        hello_cuda_from_gpu<<<GRID_DIM, BLOCK_DIM>>>();
        cudaDeviceSynchronize();        // 同步CPU主机端和GPU设备端
    }
    printf("\n");
    // CPU: 你好, CUDA! (C++版)
    {
        for(int i = 0; i < 2 * 8; ++i) {
            hello_cuda_from_cpu();
        }
    }
    return 0;
}