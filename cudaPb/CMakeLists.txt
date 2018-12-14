cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(cudaPb)

include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}/../common/masks/
	)
     
find_package(CUDA)

cuda_add_library(cudaPb
  	cudapbdetector.cu
  	cudapbdetector.h
	cvector.h
	cvector.cpp
	histoallocator.h
	histoallocator.cu
	STATIC
  )

target_link_libraries(cudaPb libCommon ${CUDA_LIBRARIES})
target_compile_features(cudaPb PUBLIC cxx_std_11)
 
 
