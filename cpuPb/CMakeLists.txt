cmake_minimum_required (VERSION 3.10)
project (cpuPb)
cmake_policy(SET CMP0020 NEW)

include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}/../common/masks/
	${OpenCV_INCLUDE_DIRS})

set(MPB_SRCS 	
	pbdetector.cpp
)

set(MPB_HEADR
	pbdetector.h)

add_library(cpuPb  STATIC ${MPB_SRCS} ${MPB_HEADR})
target_link_libraries(cpuPb
	libCommon
	${OpenCV_LIBS})
	
