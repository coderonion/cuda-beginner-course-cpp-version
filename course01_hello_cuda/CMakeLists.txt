cmake_minimum_required(VERSION 3.22)
project(course01_hello_cuda)

# 设置GPU架构/CUDA计算能力版本号
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)
# add_definitions(-arch=sm_70)

# Linux系统配置nvcc编译器
if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
endif()
# 开启CUDA编译器支持
enable_language(CUDA)

# 开启C++17语言标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# 设置相关编译选项
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# 使用指定的源文件来生成目标可执行文件
add_executable(course01_hello_cuda course01_hello_cuda.cu)
# 开启CUDA的可分离编译特性
set_target_properties(course01_hello_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)