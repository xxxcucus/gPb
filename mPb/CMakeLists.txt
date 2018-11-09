cmake_minimum_required (VERSION 3.10)
project (mPb)
cmake_policy(SET CMP0020 NEW)

include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}/../common
	${Qt5Core_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS})

set(MPB_SRCS 	
	main.cpp
	pbdetector.cpp
)

set(MPB_HEADR
	pbdetector.h)

add_executable(mPb ${MPB_SRCS} ${MPB_HEADR})
target_link_libraries(mPb
	libCommon
	${OpenCV_LIBS})
	
qt5_use_modules(mPb Core)
