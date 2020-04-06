//
// Created by 许斌 on 2020/4/3.
//
#include <iostream>
#include <opencv2/opencv.hpp>
#include "face_detector/face_detector.h"
using namespace face_detector;
using namespace n_config;
using namespace std;
using namespace cv;
using namespace Eigen;

FaceDetector::FaceDetector()
{
    config_ = new NConfig();
    config_->init("../config/config.txt");
}

FaceDetector::~FaceDetector()
{
    delete config_;
}

void FaceDetector::init(const std::string config_path)
{
    config_->init(config_path);
}

void FaceDetector::process()
{
    int size = config_->image_width_ * config_->image_height_;  // 未进行pca处理图像维数
    // 将图像转化为列向量并存储
    setTrainMatrix(config_->train_path_);
    setTestMatrix(config_->test_path_);

    cout << "Finished Matrix Setting..." << endl;

    // 计算每一维的均值`
    MatrixXd mean_Matrix_ = train_Matrix_.colwise().mean();
    train_mean = mean_Matrix_;

    // 零均值化
    MatrixXd train_zero_Matrix = train_Matrix_;
    train_zero_Matrix.rowwise() -= train_mean;

    cout << "Finished Zero..." << endl;

    // 求协方差矩阵
    MatrixXd C = train_zero_Matrix * train_zero_Matrix.transpose() / size;

    cout << "Finished compute Cov..." << endl;

    // 计算协方差矩阵的特征值和特征向量，使用selfadjont可以让产生的特征值和特征向量有序排列
    SelfAdjointEigenSolver<MatrixXd> eigen_solver(C);
    MatrixXd vec = eigen_solver.eigenvectors().rightCols(config_->dim_);  // 取前dim_列特征向量
//    MatrixXd val = eigen_solver.eigenvalues();    // 特征值

    basic_Matrix_ = train_zero_Matrix.transpose() * vec;

    MatrixXd basic_square_Matrix = basic_Matrix_.array().square();

    for(int i = 0; i < basic_Matrix_.cols(); i++)
    {
        basic_Matrix_.col(i) = basic_Matrix_.col(i) / sqrt(basic_square_Matrix.col(i).sum());
    }
    train_data_Matrix_ = train_zero_Matrix * basic_Matrix_;
}

void FaceDetector::detector()
{
    int detector_label;
    MatrixXd test_zero_Matrix = test_Matrix_;
    test_zero_Matrix.rowwise() -= train_mean;

    test_data_Matrix_ = test_zero_Matrix * basic_Matrix_;

    int count = 0;
    for(int i = 0; i < test_data_Matrix_.rows(); i++)
    {
        double min_difference = -1;
        for(int j = 0; j < train_data_Matrix_.rows(); j++)
        {
            double difference = (test_data_Matrix_.row(i) - train_data_Matrix_.row(j)).squaredNorm();
            if(min_difference < 0 || difference < min_difference)
            {
                min_difference = difference;
                detector_label = train_label_[j];
            }
        }
        if(test_label_[i] == detector_label)
        {
            count++;
        }
        cout << "识别结果：" << detector_label << "    测试标签：" << test_label_[i] << endl;
    }
    cout << "准确率：" << count * 1.0 / test_data_Matrix_.rows() * 100 << "%" << endl;
}

void FaceDetector::setTrainMatrix(const string train_path_)
{
    if(train_path_.size() == 0) cout << "The Path is empty..." << endl;
    fstream file(train_path_);
    if(!file.is_open()) cout << "Open File Failure..." << endl;

    string content;
    int size = config_->image_height_ * config_->image_width_;
    vector<Mat> image_data;
    while (getline(file, content))
    {
        vector<string> sp = config_->n_split(content, ' ');
        Mat image = imread(sp[0], IMREAD_GRAYSCALE);
        image_data.push_back(image.reshape(0, 1));
        train_label_.push_back(stoi(sp[1]));
    }
    train_Matrix_ = MatrixXd::Zero(image_data.size(), size);
    for(int i = 0; i < image_data.size(); i++)
    {
        for(int j = 0; j < size; j++)
        {
            train_Matrix_.coeffRef(i, j) = image_data[i].data[j];
        }
    }
}

void FaceDetector::setTestMatrix(const string test_path_)
{
    if(test_path_.size() == 0) cout << "The Path is Empty..." << endl;
    fstream file(test_path_);
    if(!file.is_open()) cout << "Open File Failure..." << endl;

    string content;
    int size = config_->image_height_ * config_->image_width_;
    vector<Mat> image_data;
    while (getline(file, content))
    {
        vector<string> sp = config_->n_split(content, ' ');
        Mat image = imread(sp[0], IMREAD_GRAYSCALE);
        image_data.push_back(image.reshape(0, 1));
        test_label_.push_back(stoi(sp[1]));
    }
    test_Matrix_ = MatrixXd::Zero(image_data.size(), size);
    for(int i = 0; i < image_data.size(); i++)
    {
        for(int j = 0; j < size; j++)
        {
            test_Matrix_.coeffRef(i, j) = image_data[i].data[j];
        }
    }
}

void FaceDetector::saveData(const string save_path_) {}

void FaceDetector::readData(const string data_path_) {}