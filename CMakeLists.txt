cmake_minimum_required(VERSION 3.14)
project(rho_filter)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Eigen3 REQUIRED) # Use at least 3.3, but need to be arbitary to allow unit tests
include_directories(include ${Eigen3_INCLUDE_DIRS})

add_library(rho_filter_lib src/rhoFilter.cpp)
target_link_libraries(rho_filter_lib Eigen3::Eigen)
target_include_directories(rho_filter_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Detect ROS 2
find_package(ament_cmake QUIET)

if(ament_cmake_FOUND)
    message(STATUS "ROS 2 found. Building for ROS 2.")
    find_package(rclcpp REQUIRED)
    find_package(std_msgs REQUIRED)
    find_package(eigen3_cmake_module REQUIRED)

    add_executable(rho_filter_node src/rho_node.cpp)
    target_link_libraries(rho_filter_node rho_filter_lib)
    ament_target_dependencies(rho_filter_node rclcpp std_msgs)

    install(TARGETS rho_filter_lib rho_filter_node
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib/${PROJECT_NAME}
    )
    install(DIRECTORY include/ DESTINATION include)

    ament_package()

else()
    # === MAC/STANDALONE BUILD ===
    message(STATUS "ROS 2 not found. Building standalone test.")
    
    add_executable(manual_test test/manual_test.cpp)
    target_link_libraries(manual_test rho_filter_lib)

    # --- TRICK FOR INTELLISENSE ---
    # Create a dummy target so rho_node.cpp gets added to compile_commands.json
    # We use EXCLUDE_FROM_ALL so 'make' doesn't actually try to compile it (which would fail without ROS)
    add_library(node_intellisense_dummy OBJECT src/rho_node.cpp)
    target_link_libraries(node_intellisense_dummy rho_filter_lib Eigen3::Eigen)
    set_target_properties(node_intellisense_dummy PROPERTIES EXCLUDE_FROM_ALL TRUE)
  endif()