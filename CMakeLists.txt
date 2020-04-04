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

add_executable(face_detector demo/demo.cpp src/face_detector.cpp include/face_detector/n_config.h src/n_config.cpp)
target_link_libraries(face_detector ${OpenCV_LIBS})