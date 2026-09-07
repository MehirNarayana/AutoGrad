set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
cmake_minimum_required(VERSION 4.1)

project(AutoGrad LANGUAGES CXX)

option(AUTOGRAD_BUILD_EXAMPLES "Build AutoGrad examples" OFF)

add_subdirectory(src)

if(AUTOGRAD_BUILD_EXAMPLES)
    add_subdirectory(Examples/MNIST)
endif()
