cmake_minimum_required (VERSION 3.10)
project (gPb)

cmake_policy(SET CMP0020 NEW)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTORCC ON)

find_package(Qt5Core REQUIRED)
find_package(OpenCV 3.4.3 REQUIRED COMPONENTS core imgproc highgui imgcodecs)

if(OpenCV_FOUND)
   message("Found OpenCV")
   message("Includes: " ${OpenCV_INCLUDE_DIRS})
   message("Libs " ${OpenCV_LIBS})
else(OpenCV_FOUND)
	message("OpenCV not found" FATAL)
endif(OpenCV_FOUND)

SET(CMAKE_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(cpuPb)
add_subdirectory(Textons)
add_subdirectory(common)
add_subdirectory(cudaPb)
add_subdirectory(testMPb)
add_subdirectory(mPb)
add_subdirectory(MPbGUI)



