cmake_minimum_required (VERSION 3.10)
project (TextonGenerator)
cmake_policy(SET CMP0020 NEW)

include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}/../common/textons
	${CMAKE_CURRENT_SOURCE_DIR}/../common/
	${Qt5Core_INCLUDE_DIRS}
	${OpenCV_INCLUDES})

set(TEXTONS_SRCS 	
	main.cpp
)

add_executable(TextonGenerator 
	${TEXTONS_SRCS})
target_link_libraries(TextonGenerator
	libCommon
	${OpenCV_LIBS})
qt5_use_modules(TextonGenerator Core)

install(TARGETS TextonGenerator DESTINATION .)