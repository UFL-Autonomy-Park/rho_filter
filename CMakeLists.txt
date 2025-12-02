cmake_minimum_required(VERSION 3.10)
project(RhoFilter)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 1. Build Settings
# Force Release mode for accurate benchmarking
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Optimization Flags
if(MSVC)  # Microsoft Windows only
    add_compile_options(/O2 /Ob2)
else()
    add_compile_options(-O3 -march=native) # Target architecture
endif()

# 2. Dependencies (Eigen)
find_package(Eigen3 3.3 QUIET)

if(NOT Eigen3_FOUND)
    # Common paths for Mac/Linux
    include_directories("/opt/homebrew/include/eigen3")
    include_directories("/usr/local/include/eigen3")
    include_directories("/usr/include/eigen3")
    # Local fallback
    if(EXISTS "${CMAKE_SOURCE_DIR}/eigen")
        include_directories("${CMAKE_SOURCE_DIR}/eigen")
    endif()
else()
    include_directories(${EIGEN3_INCLUDE_DIR})
endif()

# Global Include Path (allows #include "rho_filter/rhoFilter.hpp")
include_directories(include)

# 3. Targets

# Target A: The Physics/Math Library (Native C++)
# We DO NOT include bindings.cpp here because it requires Emscripten headers.
add_library(rho_filter_lib STATIC src/rhoFilter.cpp)

# Target B: The Benchmark Executable
add_executable(bench_cpp test/benchmark.cpp)
target_link_libraries(bench_cpp rho_filter_lib)