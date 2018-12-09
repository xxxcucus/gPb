cmake_minimum_required (VERSION 3.10)
project (mPb)
cmake_policy(SET CMP0020 NEW)

find_package(CUDA)

include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}/../common/masks/
	${CMAKE_CURRENT_SOURCE_DIR}/../common/textons/
	${CMAKE_CURRENT_SOURCE_DIR}/../cudaPb
	${CMAKE_CURRENT_SOURCE_DIR}/../cpuPb
	${OpenCV_INCLUDE_DIRS}
	${CUDA_INCLUDE_DIRS})

set(MPB_SRCS 	
	mPb.cpp
)

set(MPB_HEADR 	
	mPb.h
)

cuda_add_library(mPb ${MPB_SRCS} ${MPB_HEADR} SHARED)
target_link_libraries(mPb
	libCommon
	${OpenCV_LIBS}
	cudaPb
	cpuPb)
	
