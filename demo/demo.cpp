//
// Created by 许斌 on 2020/4/8.
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
    face_detector->process();
    face_detector->detector();
    face_detector->detector("../orl_faces/s1/5.pgm", false);
}
