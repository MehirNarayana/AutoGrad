set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
cmake_minimum_required(VERSION 4.1)

project(AutoGrad LANGUAGES CXX)

# Add the library directory
add_subdirectory(src)

# Add the tests directory
add_subdirectory(testing)
