cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(cudaMPb)

include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}/../common
	)
     
find_package(CUDA)

cuda_add_library(cudaMPbTools 
  	cudampb.cu
  	cudampb.h
	cvector.h
	cvector.cpp
	STATIC
  )

target_link_libraries(cudaMPb libCommon ${CUDA_LIBRARIES})
target_compile_features(cudaMPb PUBLIC cxx_std_11)
 
 
