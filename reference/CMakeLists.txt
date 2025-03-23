cmake_minimum_required(VERSION 3.15)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

project(example)

include(ExternalProject)
include(ProcessorCount)

ProcessorCount(N)

ExternalProject_Add(
    rebound
    GIT_REPOSITORY https://github.com/hannorein/rebound.git
    GIT_TAG        4.4.6
    GIT_SHALLOW    TRUE
    CONFIGURE_COMMAND ""
    BUILD_COMMAND OPENMP=1 make librebound -j${N}
    BUILD_IN_SOURCE TRUE
    INSTALL_COMMAND mv src/librebound.so ${CMAKE_CURRENT_BINARY_DIR}
)
ExternalProject_Get_Property(rebound SOURCE_DIR)

find_package(OpenMP REQUIRED)

add_executable(example main.c)
add_dependencies(example rebound)

set(LIBRARIES m OpenMP::OpenMP_C)

target_include_directories(example PRIVATE ${SOURCE_DIR}/src)
target_link_libraries(example ${LIBRARIES} ${CMAKE_CURRENT_BINARY_DIR}/librebound.so)
