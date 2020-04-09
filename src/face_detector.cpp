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

    cout << "Finished Matrix Setting..." << endl;

    // 计算每一维的均值`
    MatrixXd mean_Matrix_ = train_Matrix_.colwise().mean();
    train_mean_ = mean_Matrix_;

    // 零均值化
    MatrixXd train_zero_Matrix = train_Matrix_;
    train_zero_Matrix.rowwise() -= train_mean_;

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
    saveData(config_->save_path_);
}

void FaceDetector::detector()
{
    setTestMatrix(config_->test_path_);
    int detector_label;
    MatrixXd test_zero_Matrix = test_Matrix_;
    test_zero_Matrix.rowwise() -= train_mean_;

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
    cout << "测试集样本数：" << test_data_Matrix_.rows() << "   识别正确数：" << count << "   准确率：" << count * 1.0 / test_data_Matrix_.rows() * 100 << "%" << endl;
}

void FaceDetector::detector(Mat image, bool is_read)
{
    int size = config_->image_height_ * config_->image_width_;
    image.reshape(0, 1);
    MatrixXd detector_Matrix = MatrixXd::Zero(1, size);
    for(int i = 0; i < size; i++)
    {
        detector_Matrix.coeffRef(1, i) = image.data[i];
    }

    MatrixXd detector_zero_Matrix = detector_Matrix;
    detector_zero_Matrix -= train_mean_;

    MatrixXd detector_data_Matrix = detector_zero_Matrix * basic_Matrix_;

    int index = 0;

    double min_difference = -1;
    for(int j = 0; j < train_data_Matrix_.rows(); j++)
    {
        double difference = (detector_data_Matrix - train_data_Matrix_.row(j)).squaredNorm();
        if(min_difference < 0 || difference < min_difference)
        {
            min_difference = difference;
            index = j;
        }
    }
    if(!is_read)
    {
        imshow("detector_image", image);
        Mat match_image = Mat::zeros(image.rows, image.cols, image.type());
        match_image.reshape(0, 1);
        for(int i = 0; i < size; i++)
        {
            match_image.data[i] = train_Matrix_.coeffRef(index, i);
        }
        match_image.reshape(0, config_->image_height_);
        imshow("match_image", match_image);
    }
    cout << "识别结果：" << getLabel(config_->label_path_, train_label_[index]) << endl;
    waitKey(0);
}

void FaceDetector::detector(const string image_path, bool is_read)
{
    if(image_path.size() <= 0)
    {
        cout << "The Image Path is Error..." << endl;
        return;
    }
    Mat image = imread(image_path, IMREAD_GRAYSCALE);
    if(image.empty())
    {
        cout << "Open Image Failure..." << endl;
        return;
    }
    cout << "识别文件：" << image_path << endl;
    detector(image, is_read);
}

void FaceDetector::read_detector()
{
    readData(config_->save_path_);
}

void FaceDetector::setTrainMatrix(const string train_path_)
{
    if(train_path_.size() <= 0) cout << "The Path is empty..." << endl;
    ifstream file(train_path_);
    if(!file.is_open()) cout << "Open File Failure..." << endl;

    string content;
    int size = config_->image_height_ * config_->image_width_;
    vector<Mat> train_image_data;
    while (getline(file, content))
    {
        vector<string> sp = config_->n_split(content, ' ');
        Mat image = imread(sp[0], IMREAD_GRAYSCALE);
        train_image_data.push_back(image.reshape(0, 1));
        train_label_.push_back(stoi(sp[1]));
    }
    train_Matrix_ = MatrixXd::Zero(train_image_data.size(), size);
    for(int i = 0; i < train_image_data.size(); i++)
    {
        for(int j = 0; j < size; j++)
        {
            train_Matrix_.coeffRef(i, j) = train_image_data[i].data[j];
        }
    }
    file.close();
}

void FaceDetector::setTestMatrix(const string test_path_)
{
    if(test_path_.size() <= 0) cout << "The Path is Empty..." << endl;
    ifstream file(test_path_);
    if(!file.is_open()) cout << "Open File Failure..." << endl;

    string content;
    int size = config_->image_height_ * config_->image_width_;
    vector<Mat> test_image_data;
    while (getline(file, content))
    {
        vector<string> sp = config_->n_split(content, ' ');
        Mat image = imread(sp[0], IMREAD_GRAYSCALE);
        test_image_data.push_back(image.reshape(0, 1));
        test_label_.push_back(stoi(sp[1]));
    }
    test_Matrix_ = MatrixXd::Zero(test_image_data.size(), size);
    for(int i = 0; i < test_image_data.size(); i++)
    {
        for(int j = 0; j < size; j++)
        {
            test_Matrix_.coeffRef(i, j) = test_image_data[i].data[j];
        }
    }
    file.close();
}

string FaceDetector::getLabel(const string label_path, int detector_label)
{
    if(label_path.size() <= 0)
    {
        return "Path Error...";
    }
    ifstream file(label_path);
    if(!file.is_open())
    {
        return "Open Label File Failure...";
    }
    string input_str;
    while (getline(file, input_str))
    {
        vector<string> sp = config_->n_split(input_str, ':');
        if(detector_label == stoi(sp[0]))
        {
            return sp[1];
        }
    }
    file.close();
    return "Can't detector...";
}

void FaceDetector::n_resize(Mat &image) {}

void FaceDetector::saveData(const string data_path_)
{
    cout << "Saving Data..." << endl;
    if(data_path_.size() <= 0)
    {
        cout << "Path Error..." << endl;
        return;
    }
    ofstream file(data_path_);
    if(!file.is_open())
    {
        cout << "Open File Failure..." << endl;
        return;
    }
    file.clear();

//    file << "train_Matrix : ";
//    for(int i = 0; i < train_Matrix_.rows(); i++)
//    {
//        file << train_Matrix_.row(i) << ",";
//    }
//    file << endl;

    file << "train_mean_Matrix : " << train_mean_ << endl;

    file << "train_label : ";
    for(int i = 0; i < train_label_.size(); i++)
    {
        file << train_label_[i] << " ";
    }
    file << endl;

    file << "train_data_Matrix : ";
    for(int i = 0; i < train_data_Matrix_.rows(); i++)
    {
        file << train_data_Matrix_.row(i) << ",";
    }
    file << endl;

    file << "basic_Matrix : ";
    for(int i = 0; i < basic_Matrix_.rows(); i++)
    {
        file << basic_Matrix_.row(i) << ",";
    }
    cout << "Save Data Success..." << endl;
    file.close();
}

void FaceDetector::readData(const string data_path_)
{
    cout << "Read Data..." << endl;
    if(data_path_.size() <= 0)
    {
        cout << "Path Error..." << endl;
        return;
    }
    ifstream file(data_path_);
    if(!file.is_open())
    {
        cout << "Open Label File Failure..." << endl;
        return;
    }
    string input_str;
    while (getline(file, input_str))
    {
        vector<string> sp = config_->n_split(input_str, ':');
//        if(sp[0] == "train_Matrix")
//        {
//            vector<string> matrix_row = config_->n_split(sp[1], ',');
//            vector<vector<double>> train_data;
//            for(int i = 0; i < matrix_row.size(); i++)
//            {
//                vector<double> vector_col;
//                vector<string> matrix_col = config_->n_split(matrix_row[i], ' ');
//                for(int j = 0; j < matrix_col.size(); j++)
//                {
//                    if(matrix_col[j] == "") continue;
//                    vector_col.push_back(stod(matrix_col[j]));
//                }
//                if(vector_col.size() > 0)   train_data.push_back(vector_col);
//            }
//            train_Matrix_ = MatrixXd::Zero(train_data.size(), train_data[0].size());
//            for(int i = 0; i < train_data.size(); i++)
//            {
//                for(int j = 0; j < train_data[i].size(); j++)
//                {
//                    train_Matrix_.coeffRef(i, j) = train_data[i][j];
//                }
//            }
//        }
        if(sp[0] == "train_mean_Matrix")
        {
            vector<string> matrix = config_->n_split(sp[1], ' ');
            vector<double> mean;
            for(int i = 0; i < matrix.size(); i++)
            {
                if(matrix[i] == "") continue;
                mean.push_back(stod(matrix[i]));
            }
            train_mean_ = MatrixXd::Zero(1, mean.size());
            for(int i = 0; i < mean.size(); i++)
            {
                train_mean_.coeffRef(0, i) = mean[i];
            }
        }
        else if(sp[0] == "train_label")
        {
            vector<string> matrix = config_->n_split(sp[1], ' ');
            for(auto it : matrix)
            {
                train_label_.push_back(stoi(it));
            }
        }
        else if(sp[0] == "train_data_Matrix")
        {
            vector<string> matrix_row = config_->n_split(sp[1], ',');
            vector<vector<double>> train_data;
            for(int i = 0; i < matrix_row.size(); i++)
            {
                vector<double> vector_col;
                vector<string> matrix_col = config_->n_split(matrix_row[i], ' ');
                for(int j = 0; j < matrix_col.size(); j++)
                {
                    if(matrix_col[j] == "") continue;
                    vector_col.push_back(stod(matrix_col[j]));
                }
                if(vector_col.size() > 0)   train_data.push_back(vector_col);
            }
            train_data_Matrix_ = MatrixXd::Zero(train_data.size(), train_data[0].size());
            for(int i = 0; i < train_data.size(); i++)
            {
                for(int j = 0; j < train_data[i].size(); j++)
                {
                    train_data_Matrix_.coeffRef(i, j) = train_data[i][j];
                }
            }
        }
        else if(sp[0] == "basic_Matrix")
        {

            vector<string> matrix_row = config_->n_split(sp[1], ',');
            vector<vector<double>> basic_data;
            for(int i = 0; i < matrix_row.size(); i++)
            {
                vector<double> vector_col;
                vector<string> matrix_col = config_->n_split(matrix_row[i], ' ');
                for(int j = 0; j < matrix_col.size(); j++)
                {
                    if(matrix_col[j] == "") continue;
                    vector_col.push_back(stod(matrix_col[j]));
                }
                if(vector_col.size() > 0)   basic_data.push_back(vector_col);
            }
            basic_Matrix_ = MatrixXd::Zero(basic_data.size(), basic_data[0].size());
            for(int i = 0; i < basic_data.size(); i++)
            {
                for(int j = 0; j < basic_data[i].size(); j++)
                {
                    basic_Matrix_.coeffRef(i, j) = basic_data[i][j];
                }
            }
        }
    }
}