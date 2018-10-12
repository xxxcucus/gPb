cmake_minimum_required (VERSION 3.10)
project (Textons)
cmake_policy(SET CMP0020 NEW)

include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}/../common
	${Qt5Core_INCLUDE_DIRS}
	${OpenCV_INCLUDES})

set(TEXTONS_SRCS 	
	gaussianfilter.cpp
	gaussianfirstderivativefilter.cpp
	gaussiansecondderivativefilter.cpp
	main.cpp
	textongenerator.cpp
	textonkernel.cpp
)

add_executable(Textons 
	${TEXTONS_SRCS})
target_link_libraries(Textons
	libCommon
	${OpenCV_LIBS})
qt5_use_modules(Textons Core)

