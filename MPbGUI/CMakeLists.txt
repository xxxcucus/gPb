cmake_minimum_required (VERSION 3.10)
project (MPbGUI)

cmake_policy(SET CMP0020 NEW)
cmake_policy(SET CMP0043 NEW)

include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}/../common
	${Qt5Widgets_INCLUDE_DIRS})

set(MPB_GUI_SRCS 	
	main.cpp
	mpbguimainwindow.cpp
	mpbguicentralwidget.cpp
)

set(MPB_GUI_HEADR
	mpbguimainwindow.h
	mpbguicentralwidget.h
)

add_executable(MPbGUI 
	${MPB_GUI_SRCS}
	${MPB_GUI_HEADR})
target_link_libraries(MPbGUI
	libCommon
	${Qt5Widgets_LIBRARIES}
	${OpenCV_LIBS})

qt5_use_modules(MPbGUI Widgets)
install(TARGETS MPbGUI DESTINATION .)