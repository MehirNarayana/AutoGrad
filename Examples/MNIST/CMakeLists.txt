

cmake_minimum_required(VERSION 4.1)
project(examples LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

file(GLOB TEST_SOURCE_FILES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_executable(${PROJECT_NAME} ${TEST_SOURCE_FILES})

set(MNIST_DATA_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/dataset")
add_custom_target(
    mnist_dataset
    COMMAND
        ${CMAKE_COMMAND}
        -DMNIST_DATA_DIRECTORY=${MNIST_DATA_DIRECTORY}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/DownloadMnist.cmake
    COMMENT "Ensuring the MNIST dataset is available"
    VERBATIM
)

add_dependencies(${PROJECT_NAME} mnist_dataset)
target_compile_definitions(
    ${PROJECT_NAME}
    PRIVATE AUTOGRAD_MNIST_DATA_DIRECTORY="${MNIST_DATA_DIRECTORY}"
)
target_link_libraries(${PROJECT_NAME} PRIVATE AutoGrad)
