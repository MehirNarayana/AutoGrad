cmake_minimum_required(VERSION 4.1)
project(AutoGrad LANGUAGES CXX)

# Add the library directory
add_subdirectory(src)

# Add the tests directory
add_subdirectory(testing)
