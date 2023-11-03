cmake_minimum_required(VERSION 3.18)
project(main)

find_package(Python3 COMPONENTS Interpreter Development NumPy)
find_package(CUDA REQUIRED)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
link_libraries(${Python3_LIBRARIES})
include_directories(${Python3_INCLUDE_DIRS})

set(CMAKE_PREFIX_PATH /home/astokely/Downloads/libtorch)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
add_executable(lltm ${CMAKE_SOURCE_DIR}/lltm/lltm.cpp
        lltm/lltm_kernel.cu)
target_link_libraries(lltm "${TORCH_LIBRARIES}")

