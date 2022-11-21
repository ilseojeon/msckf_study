cmake_minimum_required(VERSION 3.0.2)
project(homework LANGUAGES CXX VERSION 0.0.0)

cmake_policy(SET CMP0015 NEW)
cmake_policy(SET CMP0064 NEW)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -O3 -w")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)

add_executable(${PROJECT_NAME}_run homework.cpp)
target_link_libraries(${PROJECT_NAME}_run
            ${EIGEN_LIBS})
target_include_directories(${PROJECT_NAME}_run PUBLIC
            ${EIGEN_INCLUDE_DIRS}
            )