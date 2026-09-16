cmake_minimum_required(VERSION 3.27)
project(falconnpp)

set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Boost 1.71 REQUIRED NO_MODULE)
find_package(OpenMP)

add_executable(falconnpp
        src/main.cpp
        src/bf.cpp
        src/utilities.cpp
        src/falconnpp.cpp
        src/fast_copy.c
        src/fht.c
)

target_link_libraries(falconnpp PRIVATE Eigen3::Eigen)

if(OpenMP_CXX_FOUND)
    target_link_libraries(falconnpp PRIVATE OpenMP::OpenMP_CXX)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(falconnpp PRIVATE -march=native)
elseif(MSVC)
    target_compile_options(falconnpp PRIVATE /arch:AVX2)
endif()