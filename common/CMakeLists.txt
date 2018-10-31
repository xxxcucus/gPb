cmake_minimum_required (VERSION 3.10)
project (libCommon)
cmake_policy(SET CMP0020 NEW)

include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}
	${Qt5Core_INCLUDE_DIRS}
	)

set(COMMON_SRCS texton.cpp discdirectmasks.cpp discinversemasks.cpp)
add_library(libCommon STATIC ${COMMON_SRCS})

qt5_use_modules(libCommon Core)