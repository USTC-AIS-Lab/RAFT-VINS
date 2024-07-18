cmake_minimum_required(VERSION 2.8)
project(libtorch)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -funroll-loops")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto")

set(Torch_DIR /home/zj/libtorch/share/cmake/Torch)
# set(OpenCV_DIR /usr/local)

find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})

# find_package(Boost REQUIRED)
# include_directories(${Boost_INCLUDE_DIRS})
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

message("CUDA_INCLUDE_DIRS: " ${CUDA_INCLUDE_DIRS})
message("CMAKE_SOURCE_DIR: " ${CMAKE_SOURCE_DIR})
message("PROJECT_SOURCE_DIR: " ${PROJECT_SOURCE_DIR})

# message("${CMAKE_SOURCE_DIR}/src:" ${CMAKE_SOURCE_DIR}/src) # 输出：/home/zj/workspace/RAFT_CPP/src:/home/zj/workspace/RAFT_CPP/src

#-------------------------------------------------------TensorRT-------------------------------------------
# find_package(TensorRT REQUIRED)
# include_directories(${TensorRT_INCLUDE_DIRS})
# find_package(TensorRT REQUIRED)
# include_directories(${TensorRT_INCLUDE_DIRS})
set(TensorRT_INCLUDE_DIRS 
    "/home/zj/TensorRT-8.4.1.5/include"
    "/home/zj/TensorRT-8.4.1.5/samples/common"
)
# set(TensorRT_LIBRARIES 
#     "/home/zj/TensorRT-8.6.1.6/lib"
# )
include_directories(${TensorRT_INCLUDE_DIRS})
file(GLOB TensorRT_LIBRARIES "/home/zj/TensorRT-8.4.1.5/lib/*.so")
#-------------------------------------------------------TensorRT-------------------------------------------

find_package(OpenCV 4 REQUIRED)
message(${OpenCV_VERSION})
include_directories(${OpenCV_INCLUDE_DIRS})

SET(BUILD_SHARED_LIBS ON)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

# file(GLOB_RECURSE all_include_files RELATIVE "${CMAKE_SOURCE_DIR}" *.h *.hpp)

# link_directories(${CMAKE_SOURCE_DIR}/src/)
# file(GLOB_RECURSE all_source_files RELATIVE "${CMAKE_SOURCE_DIR}src/" *.cpp)

# message("all_source_files" ${all_source_files})
include_directories(${PROJECT_SOURCE_DIR}/include)
add_library(optical_flow ${CMAKE_SOURCE_DIR}/src/Pipeline.cpp 
                         ${CMAKE_SOURCE_DIR}/src/Raft_torch.cpp 
                         ${CMAKE_SOURCE_DIR}/src/utils.cpp 
                         ${CMAKE_SOURCE_DIR}/src/Visual.cpp
                         ${CMAKE_SOURCE_DIR}/src/calcflow_raft.cpp
                         ${CMAKE_SOURCE_DIR}/src/logger.cpp
                         )
target_link_libraries(optical_flow ${TORCH_LIBRARIES} ${CUDA_LIBRARIES} ${OpenCV_LIBRARIES} ${TensorRT_LIBRARIES})
# aux_source_directory(src src_source)
# add_executable(test main.cpp ${src_source})
# add_executable(test main.cpp)
# target_link_libraries(test optical_flow)
# target_link_libraries(test ${TORCH_LIBRARIES} ${CUDA_LIBRARIES} ${OpenCV_LIBRARIES})
