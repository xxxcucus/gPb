cmake_minimum_required (VERSION 2.6)
project (mPb)

include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}/../common
	${Qt5Core_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS})

set(MPB_SRCS 	
	discdirectmasks.cpp
	discinversemasks.cpp
	main.cpp
	pbdetector.cpp
)

add_executable(mPb ${MPB_SRCS})
target_link_libraries(mPb
	libCommon
	${OpenCV_LIBS})
	
qt5_use_modules(mPb Core)
