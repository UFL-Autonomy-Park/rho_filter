cmake_minimum_required(VERSION 3.10)
project(RhoFilterBenchmark)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Force Release Build Type
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler-specific optimization flags
if(MSVC)
    add_compile_options(/O2 /Ob2 /Oi /Ot)
else()
    # -O3: Max optimization
    # -march=native: Tune for local CPU architecture (SIMD)
    add_compile_options(-O3 -march=native)
endif()

find_package(Eigen3 3.3 QUIET)

if(NOT Eigen3_FOUND)
    # Search common Homebrew/Linux paths
    include_directories("/opt/homebrew/include/eigen3")
    include_directories("/usr/local/include/eigen3")
    include_directories("/usr/include/eigen3")
    # Windows fallback (if cloned locally as per README)
    include_directories("${CMAKE_SOURCE_DIR}/eigen")
else()
    include_directories(${EIGEN3_INCLUDE_DIR})
endif()

add_executable(bench_cpp benchmark.cpp)