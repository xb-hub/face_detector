cmake_minimum_required(VERSION 3.15)
project(face_detector)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(
        include
        ${OpenCV_INCLUDE_DIR}
        ${EIGEN3_INCLUDE_DIR}
)

add_executable(face_detector demo/demo.cpp src/face_detector.cpp src/n_config.cpp demo/detector_demo.cpp)
target_link_libraries(face_detector ${OpenCV_LIBS})

add_executable(detector_demo demo/detector_demo.cpp src/face_detector.cpp  src/n_config.cpp demo/detector_demo.cpp)
target_link_libraries(detector_demo ${OpenCV_LIBS})