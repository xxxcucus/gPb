cmake_minimum_required (VERSION 3.10)
project (Textons)
cmake_policy(SET CMP0020 NEW)

include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}/../common/textons
	${Qt5Core_INCLUDE_DIRS}
	${OpenCV_INCLUDES})

set(TEXTONS_SRCS 	
	main.cpp
	textongenerator.cpp
)

set(TEXTON_HEADR
	textongenerator.h)

add_executable(Textons 
	${TEXTONS_SRCS} ${TEXTON_HEADR})
target_link_libraries(Textons
	libCommon
	${OpenCV_LIBS})
qt5_use_modules(Textons Core)

install(TARGETS Textons DESTINATION .)