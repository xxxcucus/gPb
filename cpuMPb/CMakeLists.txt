cmake_minimum_required (VERSION 3.10)
project (cpuMPb)
cmake_policy(SET CMP0020 NEW)

#find_package(CUDA)

include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}/../common
	#${CMAKE_CURRENT_SOURCE_DIR}/../cudaMPb
	#${Qt5Core_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS})
	#${CUDA_INCLUDE_DIRS})

set(MPB_SRCS 	
	pbdetector.cpp
)

set(MPB_HEADR
	pbdetector.h)

add_library(cpuMPb  STATIC ${MPB_SRCS} ${MPB_HEADR})
target_link_libraries(cpuMPb
	libCommon
	${OpenCV_LIBS})
	#cudaMPb
	
#qt5_use_modules(cpuMPb Core)

#install(TARGETS mPb DESTINATION .)
install (FILES ${CMAKE_ROOT_DIR}/cpuMPb/Sternchen2016.jpg DESTINATION .)