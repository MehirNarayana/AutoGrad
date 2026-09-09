cmake_minimum_required(VERSION 4.1)
project(AutoGrad)

file(GLOB SOURCE_FILES CONFIGURE_DEPENDS ./*.cpp)

add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES})
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
