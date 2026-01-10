cmake_minimum_required(VERSION 3.18)

project(lab5 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

find_package(OpenCV REQUIRED)

include_directories(/usr/include/x86_64-linux-gnu /usr/local/cuda/include)

add_executable(lab5 src/main.cu)

target_link_libraries(lab5 
    ${OpenCV_LIBS}
    nvinfer
    nvinfer_plugin
    nvparsers
    cudart
)
