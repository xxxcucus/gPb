cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(cudaMPbTools LANGUAGES CXX CUDA)

include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}/../common
	)
     
add_library(cudaMPbTools STATIC
  	cudaimage.cu
  	cudaimage.h
  )

 target_link_libraries(cudaMPbTools libCommon)
 
# Request that particles be built with -std=c++11
# As this is a public compile feature anything that links to 
# particles will also build with -std=c++11
target_compile_features(cudaMPbTools PUBLIC cxx_std_11)
 
# We need to explicitly state that we need all CUDA files in the 
# particle library to be built with -dc as the member functions 
# could be called by other libraries and executables
set_target_properties( cudaMPbTools
					   PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
 
