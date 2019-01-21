cmake_minimum_required (VERSION 2.6)
project (cudnn-training)
set(HIPDNN_PATH "/opt/rocm/hipdnn" )
message(STATUS "HIPDNN_PATH " ${HIPDNN_PATH})
set(HIPBLAS_PATH "/opt/rocm/hipblas")
message(STATUS "hipblas" ${HIPBLAS_PATH})

SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake  "${HIP_PATH}/cmake")
add_definitions(-std=c++11)
# HIP_PATH
IF(NOT DEFINED ENV{HIP_PATH})
  SET(HIP_PATH "/opt/rocm/hip")
ELSE()
  SET(HIP_PATH $ENV{HIP_PATH})
ENDIF()

EXECUTE_PROCESS(COMMAND ${HIP_PATH}/bin/hipconfig -P OUTPUT_VARIABLE HIP_PLATFORM)
IF(${HIP_PLATFORM} MATCHES "hcc")
  find_package(MIOpen REQUIRED)
  include_directories(${MIOpen_INCLUDE_DIRS})
ELSE()
  find_package(CuDNN REQUIRED)
  include_directories(${CuDNN_INCLUDE_DIRS})
ENDIF()
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
SET(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS)
SET(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS)

if(USE_GFLAGS)
  add_definitions(-DUSE_GFLAGS)
endif()

SET(CMAKE_CXX_COMPILER "/opt/rocm/hip/bin/hipcc")
include_directories(${HIP_PATH}/include ${HIP_PATH} ${HIPDNN_PATH}/include ${HIPDNN_PATH} ${HIPBLAS_PATH} ${HIPBLAS_PATH}/include )
link_directories(${HIPBLAS_PATH}/lib/ ${HIPDNN_PATH}/lib/)
add_executable(trainlenet lenet.cpp readubyte.cpp)

target_link_libraries(trainlenet hipdnn hipblas)


