cmake_minimum_required(VERSION 3.14)
project(rho_filter)

set(CMAKE_CXX_STANDARD 17) # Which should I use? Latest? 17? What's best for ROS2?

# Try to find Eigen3 via Config mode (Homebrew standard)
find_package(Eigen3 3.3 QUIET CONFIG)

if(EIGEN3_FOUND)
    include_directories(${EIGEN3_INCLUDE_DIR})
else()
    # Fallback: Look in local directory if brew version isn't found
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/eigen)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    # -O3: Maximum optimization
    # -march=native: Use architecture-specific instructions (AVX/NEON) available on THIS computer
    # -DNDEBUG: Disables all assert() calls (crucial for speed benchmarking)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -DNDEBUG")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fsanitize=address -fno-omit-frame-pointer")
add_executable(rhoFilter main.cpp rhoFilter.cpp)