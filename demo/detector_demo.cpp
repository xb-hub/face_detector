//
// Created by 许斌 on 2020/4/3.
//
#include <iostream>
#include "face_detector/n_config.h"
#include "face_detector/face_detector.h"
using namespace n_config;
using namespace face_detector;
using namespace std;

int main() {
    FaceDetector *face_detector = new FaceDetector();
    face_detector->init("../config/config.txt");
    face_detector->read_detector();

    face_detector->read_detector("../orl_faces/s5/8.pgm");
}
