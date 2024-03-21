#include <cstdio>
#include <iostream>

using namespace std;

int main()
{
    // CUDA device count
    int cuda_device_count = 0;
    cudaError_t cuda_error_info = cudaGetDeviceCount(&cuda_device_count);
    if(cuda_error_info != cudaSuccess) {
        std::cout << "cudaGetDeviceCount error info: " << cuda_error_info << std::endl;
        return -1;
    } else {
        printf("Detected %d CUDA Capable device(s)\n", cuda_device_count);
    }
    for(int device_id = 0; device_id < cuda_device_count; ++device_id) {
        cuda_error_info = cudaSetDevice(device_id);
        if(cuda_error_info != cudaSuccess) {
            std::cout << "cudaSetDevice error info: " << cuda_error_info << std::endl;
            continue;
        }
        // CUDA device properties
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device_id);
        // CUDA device name
        char* device_name = prop.name;
        // CUDA device compute capability
        int device_major_compute_capability = prop.major;
        int device_minor_compute_capability = prop.minor;
        // CUDA Device Global memory available on device in bytes
        size_t device_total_global_mem = prop.totalGlobalMem;
        // Clock frequency in kilohertz
        int device_clock_rate = prop.clockRate;
        // Peak memory clock frequency in kilohertz
        int device_memory_clock_rate = prop.memoryClockRate;
        // Global memory bus width in bits
        int device_memory_bus_width = prop.memoryBusWidth;
        // Size of L2 cache in bytes
        int device_l2_cache_size = prop.l2CacheSize;
        // CUDA Device Constant memory available on device in bytes
        size_t device_total_const_mem = prop.totalConstMem;
        // Shared memory available per block in bytes
        size_t device_shared_mem_per_block = prop.sharedMemPerBlock;
        // 32-bit registers available per block
        int device_regs_per_block = prop.regsPerBlock;
        // Warp size in threads
        int device_warp_size = prop.warpSize;
        // Maximum resident threads per multiprocessor
        int device_max_threads_per_multi_processor = prop.maxThreadsPerMultiProcessor;
        // Maximum number of threads per block
        int device_max_threads_per_block = prop.maxThreadsPerBlock;
        // Maximum size of each dimension of a block
        int* device_max_threads_dim = prop.maxThreadsDim;
        // Maximum size of each dimension of a grid
        int* device_max_grid_size = prop.maxGridSize;
        printf(" Device %d: %s\n", device_id, device_name);
        printf(" CUDA Capability Major/Minor version number:    %d.%d\n", device_major_compute_capability, device_minor_compute_capability);
        printf(" Total amount of global memory:                 %.0lf MBytes (%lld bytes)\n", device_total_global_mem / (1024.0 * 1024.0), device_total_global_mem);
        printf(" GPU Max Clock rate:                            %.0f MHz\n", device_clock_rate / (1000.0));
        printf(" Memory Clock rate:                             %.0f MHz\n", device_memory_clock_rate / (1000.0));
        printf(" Memory Bus Width:                              %d-bit\n", device_memory_bus_width);
        printf(" L2 Cache Size:                                 %d bytes\n", device_l2_cache_size);
        printf(" Total amount of constant memory:               %lld bytes\n", device_total_const_mem);
        printf(" Total amount of shared memory per block:       %lld bytes\n", device_shared_mem_per_block);
        printf(" Total number of registers available per block: %d\n", device_regs_per_block);
        printf(" Warp Size:                                     %d\n", device_warp_size);
        printf(" Maximum number of threads per multiprocessor:  %d\n", device_max_threads_per_multi_processor);
        printf(" Maximum number of threads per block:           %d\n", device_max_threads_per_block);
        printf(" Max dimension size of a thread block (x,y,z):  (%d, %d, %d)\n", device_max_threads_dim[0], device_max_threads_dim[1], device_max_threads_dim[2]);
        printf(" Max dimension size of a grid size (x,y,z):     (%d, %d, %d)\n", device_max_grid_size[0], device_max_grid_size[1], device_max_grid_size[2]);
    }
    return 0;
}