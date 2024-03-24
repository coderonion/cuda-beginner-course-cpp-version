#include <iostream>
#include <chrono>

#include "array_add_and_minus_cpu.hpp"
#include "array_add_and_minus_gpu.cuh"

using namespace std::chrono;

int main() {
    // CPU版：主机端进行3个一维数组加减运算
    double * a_host, * b_host, * c_host, * d_host;
    {
        const auto t_start = steady_clock::now();

        a_host = new double[ARRAY_LEN]; // 主机端：a_host数组分配内存约为0.8GB
        b_host = new double[ARRAY_LEN]; // 主机端：b_host数组分配内存约为0.8GB
        c_host = new double[ARRAY_LEN]; // 主机端：c_host数组分配内存约为0.8GB
        d_host = new double[ARRAY_LEN]; // 主机端：d_host数组分配内存约为0.8GB
        // 主机端：a_host, b_host, c_host这3个数组分别用常量进行初始化
        for (unsigned long i = 0; i < ARRAY_LEN; ++i) {
            a_host[i] = VALUE_A;
            b_host[i] = VALUE_B;
            c_host[i] = VALUE_C;
        }

        const auto t_add_minus = steady_clock::now();

        // 主机端：a_host, b_host这2个数组的对应位置元素相加, 结果与c_host这个数组的对应元素相减，结果输出到数组d_host
        array_add_and_minus_cpu(a_host, b_host, c_host, d_host, ARRAY_LEN);

        const auto tt_add_minus = duration_cast<milliseconds>(steady_clock::now() - t_add_minus);
        const auto tt_total = duration_cast<milliseconds>(steady_clock::now() - t_start);

        std::string check_array_status = check_array_add_and_minus(d_host, ARRAY_LEN) ? "数组加减运算正确" : "数组加减运算错误";
        std::cout << "CPU: " << check_array_status \
            << ", 仅数组加减运算操作耗时" << tt_add_minus.count() << "毫秒" \
            << ", 内存分配与数据拷贝操作耗时" << tt_total.count() - tt_add_minus.count() << "毫秒" \
            << ", 总耗时" << tt_total.count() << "毫秒. (C++版)" << std::endl;
    }
    // 数组d_host元素的值清0
    memset(d_host,0,ARRAY_SIZE);

    // GPU版：设备端进行3个一维数组加减运算
    double * a_device, * b_device, * c_device, *d_device;
    {
        // 通过cudaEvent统计时间
        cudaEvent_t t_start, t_add_minus_start, t_add_minus_stop, t_stop;
        float tt_add_minus_ms = 0;
        float tt_total_ms = 0;
        cudaEventCreate(&t_start);
        cudaEventCreate(&t_add_minus_start);
        cudaEventCreate(&t_add_minus_stop);
        cudaEventCreate(&t_stop);

        cudaEventRecord(t_start);

        cudaMalloc(&a_device, ARRAY_SIZE); // 设备端:a_device数组分配内存约为0.8GB
        cudaMalloc(&b_device, ARRAY_SIZE); // 设备端:b_device数组分配内存约为0.8GB
        cudaMalloc(&c_device, ARRAY_SIZE); // 设备端:c_device数组分配内存约为0.8GB
        cudaMalloc(&d_device, ARRAY_SIZE); // 设备端:d_device数组分配内存约为0.8GB
        cudaMemcpy(a_device, a_host,ARRAY_SIZE, cudaMemcpyHostToDevice); // 将主机端数组a_host的数据拷贝到设备端a_device数组中
        cudaMemcpy(b_device, b_host,ARRAY_SIZE, cudaMemcpyHostToDevice); // 将主机端数组b_host的数据拷贝到设备端b_device数组中
        cudaMemcpy(c_device, c_host,ARRAY_SIZE, cudaMemcpyHostToDevice); // 将主机端数组c_host的数据拷贝到设备端c_device数组中

        cudaEventRecord(t_add_minus_start);

        // 指定设备端核函数所使用的总线程数为数组长度ARRAY_LEN
        const int BLOCK_DIM = 1000; // 指定每个线程块中的线程数量为1000
        const int GRID_DIM = (ARRAY_LEN + BLOCK_DIM - 1) / BLOCK_DIM; // 指定网格中的线程块数量为 = (ARRAY_LEN / BLOCK_DIM)求商的整数部分 + 1
        // 设备端：a_device, b_device这2个数组的对应位置元素相加, 结果与c_device这个数组的对应元素相减，结果输出到数组d_device
        array_add_and_minus_gpu<<<GRID_DIM, BLOCK_DIM>>>(a_device, b_device, c_device, d_device);

        cudaEventRecord(t_add_minus_stop);
        cudaEventSynchronize(t_add_minus_stop);
        cudaEventElapsedTime(&tt_add_minus_ms, t_add_minus_start, t_add_minus_stop);

        // 同步主机端和设备端
        cudaDeviceSynchronize();

        // 将设备端数组d_device的数据拷贝到主机端d_host数组中
        cudaMemcpy(d_host, d_device,ARRAY_SIZE, cudaMemcpyDeviceToHost);

        cudaEventRecord(t_stop);
        cudaEventSynchronize(t_stop);
        cudaEventElapsedTime(&tt_total_ms, t_start, t_stop);

        std::string check_array_status = check_array_add_and_minus(d_host, ARRAY_LEN) ? "数组加减运算正确" : "数组加减运算错误";
        std::cout << "GPU: " << check_array_status \
            << ", 仅数组加减运算操作耗时" << tt_add_minus_ms << "毫秒" \
            << ", 内存分配与数据拷贝操作耗时" << tt_total_ms - tt_add_minus_ms << "毫秒" \
            << ", 总耗时" << tt_total_ms << "毫秒. (C++版)" << std::endl;
    }

    delete []a_host;
    delete []b_host;
    delete []c_host;
    delete []d_host;
    cudaFree(a_device);
    cudaFree(b_device);
    cudaFree(c_device);
    cudaFree(d_device);
    return 0;
}


