cmake_minimum_required(VERSION 4.1)
project(AutoGrad)


find_package(Eigen3 REQUIRED)
file(GLOB SOURCE_FILES ./*.cpp)

add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES})
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)

target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen)







