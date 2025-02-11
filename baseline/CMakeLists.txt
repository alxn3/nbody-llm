cmake_minimum_required(VERSION 3.15)

project(baseline)

include(ExternalProject)
include(ProcessorCount)

ProcessorCount(N)

ExternalProject_Add(
    rebound
    GIT_REPOSITORY https://github.com/hannorein/rebound.git
    GIT_TAG        4.4.6
    GIT_SHALLOW    TRUE
    CONFIGURE_COMMAND ""
    BUILD_COMMAND make librebound -j${N}
    BUILD_IN_SOURCE TRUE
    INSTALL_COMMAND ""
)
ExternalProject_Get_Property(rebound SOURCE_DIR)

add_executable(baseline main.c)
add_dependencies(baseline rebound)

set(LIBRARIES m)

target_include_directories(baseline PRIVATE ${SOURCE_DIR}/src)
target_link_libraries(baseline ${LIBRARIES} ${SOURCE_DIR}/librebound.so)
