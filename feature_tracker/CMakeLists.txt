cmake_minimum_required(VERSION 2.8.3)
project(feature_tracker)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
# set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set (Torch_DIR /home/zj/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

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

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    sensor_msgs
    cv_bridge
    camera_model
    )

find_package(OpenCV 4 REQUIRED)

catkin_package()

include_directories(
    ${catkin_INCLUDE_DIRS}
    )
include_directories(/home/zj/workspace/paper_final/vins_ws_v4/src/VINS-Mono/feature_tracker/src/raftflow/include)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

add_executable(feature_tracker
    src/feature_tracker_node.cpp
    src/parameters.cpp
    src/feature_tracker.cpp
    )

target_link_libraries(feature_tracker ${catkin_LIBRARIES} ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${TensorRT_LIBRARIES} ${CUDA_LIBRARIES}

/home/zj/workspace/paper_final/vins_ws_v4/src/VINS-Mono/feature_tracker/src/raftflow/lib/liboptical_flow.so)
