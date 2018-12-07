cmake_minimum_required (VERSION 3.10)
project (testMPb)
cmake_policy(SET CMP0020 NEW)

find_package(CUDA)

include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}/../common/masks/
	${CMAKE_CURRENT_SOURCE_DIR}/../common/textons/
	${CMAKE_CURRENT_SOURCE_DIR}/../cudaPb
	${CMAKE_CURRENT_SOURCE_DIR}/../cpuPb
	${Qt5Core_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS}
	${CUDA_INCLUDE_DIRS})

set(TEST_MPB_SRCS 	
	main.cpp
)

cuda_add_executable(testMPb ${TEST_MPB_SRCS})
target_link_libraries(testMPb
	libCommon
	${OpenCV_LIBS}
	cudaPb
	cpuPb)
	
qt5_use_modules(testMPb Core)
install(TARGETS testMPb DESTINATION .)
