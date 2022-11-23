cmake_minimum_required(VERSION 3.0.2)
project(msckf_study LANGUAGES CXX VERSION 0.0.0)

cmake_policy(SET CMP0015 NEW)
cmake_policy(SET CMP0064 NEW)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -O0 -w")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)
set(EIGEN_INCLUDE "/usr/local/include/eigen3/")

add_executable(${PROJECT_NAME}_run src/run_optimizer.cpp)
target_link_libraries(${PROJECT_NAME}_run
            ${EIGEN_LIBS})
target_include_directories(${PROJECT_NAME}_run PUBLIC
            ${EIGEN_INCLUDE}
            include/
            )