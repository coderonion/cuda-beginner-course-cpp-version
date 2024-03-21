cmake_minimum_required(VERSION 3.22)
project(course02_cuda_device_properties)

# add_definitions(-arch=sm_70)
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)
if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
endif()
enable_language(CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_executable(course02_cuda_device_properties course02_cuda_device_properties.cu)
set_target_properties(course02_cuda_device_properties PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)