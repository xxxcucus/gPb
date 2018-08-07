cmake_minimum_required (VERSION 2.6)
project (libCommon)

include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}
	${Qt5Core_INCLUDE_DIRS}
	)

set(COMMON_SRCS texton.cpp)
add_library(libCommon STATIC ${COMMON_SRCS})

#target_link_libraries(PlanesWidget ${Qt5Widgets_LIBRARIES})

qt5_use_modules(libCommon Core)