cmake_minimum_required(VERSION 3.2)
project(libtorch)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

set(Torch_DIR /home/zj/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})

#-------------------------------------------------------TensorRT-------------------------------------------
set(TensorRT_INCLUDE_DIRS 
    "/home/zj/TensorRT-8.4.1.5/include"
    "/home/zj/TensorRT-8.4.1.5/samples/common"
)
include_directories(${TensorRT_INCLUDE_DIRS})

file(GLOB TensorRT_LIBRARIES "/home/zj/TensorRT-8.4.1.5/lib/*.so")
#-----------------------------------------------------------------------------------------------------------

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

find_package(catkin REQUIRED COMPONENTS 
                roscpp
                rospy
                cv_bridge
                std_msgs
                sensor_msgs
            )  
include_directories(${catkin_INCLUDE_DIRS})         

find_package(Python3 COMPONENTS Development NumPy)
# include_directories(${Python3_INCLUDE_DIRS})
# include_directories("/usr/include/python3.8")
# message("Python3_INCLUDE_DIRS: " ${Python3_INCLUDE_DIRS})

find_package(OpenCV 4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(${CMAKE_SOURCE_DIR}/include)

SET(BUILD_SHARED_LIBS ON)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

add_library(optical_flow ${CMAKE_SOURCE_DIR}/src/Pipeline.cpp 
                         ${CMAKE_SOURCE_DIR}/src/Raft_torch.cpp 
                         ${CMAKE_SOURCE_DIR}/src/utils.cpp 
                         ${CMAKE_SOURCE_DIR}/src/Visual.cpp
                         ${CMAKE_SOURCE_DIR}/src/calcflow_raft.cpp
                         ${CMAKE_SOURCE_DIR}/src/logger.cpp
                         )
target_link_libraries(optical_flow ${TORCH_LIBRARIES} ${CUDA_LIBRARIES} ${OpenCV_LIBRARIES} ${TensorRT_LIBRARIES})
add_executable(raftcpp main.cpp)
target_link_libraries(raftcpp optical_flow)
target_link_libraries(raftcpp ${TORCH_LIBRARIES} ${CUDA_LIBRARIES} ${OpenCV_LIBRARIES})

# build
set(Builde_file "${CMAKE_SOURCE_DIR}/src/logger.cpp")
# message("Builde_file: " ${Builde_file})
add_executable(buildengine build_final.cpp ${Builde_file})
# # message("src_source: " ${src_source}) src_source: src/Pipeline.cppsrc/Raft_torch.cppsrc/Visual.cppsrc/calcflow_raft.cppsrc/logger.cppsrc/utils.cpp
# target_link_libraries(buildtr ${TORCH_LIBRARIES} ${CUDA_LIBRARIES} ${TensorRT_LIBRARIES})
target_link_libraries(buildengine ${TensorRT_LIBRARIES})

# add_executable(buildorin build_orin.cpp ${Builde_file})
# target_link_libraries(buildorin ${TensorRT_LIBRARIES})
# evaluate
# add_executable(evaluate evaluate.cpp)
# target_link_libraries(evaluate ${Python3_LIBRARIES} Python3::NumPy ${catkin_LIBRARIES})


#不链接动态库
# aux_source_directory(src src_source)
# message("src_source: " ${src_source})
# add_executable(test2 main.cpp ${src_source})
# target_link_libraries(test2 ${TORCH_LIBRARIES} ${CUDA_LIBRARIES} ${OpenCV_LIBRARIES} ${TensorRT_LIBRARIES})
