cmake_minimum_required(VERSION 2.8)

project( evaluation )


set(ENV{OpenCV_DIR} /usr/local)

find_package( OpenCV REQUIRED )


message(STATUS "OpenCV library status:")
message(STATUS "OpenCV ${OpenCV_DIR}") 
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")


include_directories(
	/usr/local/include
	/usr/local
	h
	)

add_executable( 
	evaluation 
	main.cpp 
	h/eval.hpp
	src/eval.cpp
	h/alter.hpp
	src/alter.cpp
	)

target_link_libraries( 
	evaluation 
	${OpenCV_LIBS} 
	)

target_compile_features(
	evaluation 
	PRIVATE 
	cxx_range_for
	)
