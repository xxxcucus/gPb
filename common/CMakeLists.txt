cmake_minimum_required (VERSION 3.10)
project (libCommon)
cmake_policy(SET CMP0020 NEW)

include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}/common/
	${CMAKE_CURRENT_SOURCE_DIR}/textons/
	${Qt5Core_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS}
	)

set(COMMON_SRCS
	textons/textongenerator.cpp
	textons/textontools.cpp
	textons/texton.cpp
	textons/textonkernel.cpp
	textons/gaussianfilter.cpp
	textons/gaussianfirstderivativefilter.cpp
	textons/gaussiansecondderivativefilter.cpp
	masks/discdirectmasks.cpp
	masks/discinversemasks.cpp
	)

set(COMMON_HEADR 
	textons/textongenerator.h
	textons/textontools.h
	textons/texton.h
	textons/textonkernel.h
	textons/gaussianfilter.h
	textons/gaussianfirstderivativefilter.h
	textons/gaussiansecondderivativefilter.h	
	masks/discdirectmasks.h
	masks/discinversemasks.h
	)
	
add_library(libCommon STATIC ${COMMON_SRCS} ${COMMON_HEADR})
target_link_libraries(libCommon ${OpenCV_LIBS})
qt5_use_modules(libCommon Core)