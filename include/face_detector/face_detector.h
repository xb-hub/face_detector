//
// Created by 许斌 on 2020/4/3.
//

#ifndef FACE_DETECTOR_FACE_DETECTOR_H
#define FACE_DETECTOR_FACE_DETECTOR_H

#include <vector>
#include <Eigen/Dense>
#include "face_detector/n_config.h"

namespace face_detector
{
class FaceDetector
{
private:
    n_config::NConfig *config_;
    Eigen::MatrixXd train_Matrix_;
    Eigen::MatrixXd test_Matrix_;
    Eigen::MatrixXd train_data_Matrix_;
    Eigen::MatrixXd test_data_Matrix_;
    Eigen::MatrixXd basic_Matrix_;
    Eigen::RowVectorXd train_mean;

    std::vector<int> train_label_;
    std::vector<int> test_label_;

public:
    FaceDetector();
    ~FaceDetector();

    void setTrainMatrix(const std::string train_path_);
    void setTestMatrix(const std::string test_path_);

    void saveData(const std::string save_path_);
    void readData(const std::string data_path_);

    void init(const std::string config_path);
    void process();
    void detector();
};
}

#endif //FACE_DETECTOR_FACE_DETECTOR_H
