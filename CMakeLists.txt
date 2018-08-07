cmake_minimum_required (VERSION 2.6)
project (gPb)

cmake_policy(SET CMP0020 NEW)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTORCC ON)

find_package(Qt5Core REQUIRED)
find_package(OpenCV REQUIRED)

if(OpenCV_FOUND)
   message("Found OpenCV")
   message("Includes: " ${OpenCV_INCLUDE_DIRS})
else(OpenCV_FOUND)
	message("OpenCV not found" FATAL)
endif(OpenCV_FOUND)


add_subdirectory(mPb)
add_subdirectory(Textons)
add_subdirectory(common)



