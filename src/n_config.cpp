//
// Created by 许斌 on 2020/4/3.
//
#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include "face_detector/n_config.h"
using namespace std;
using namespace n_config;

// trim from start
static inline std::string &ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
    return s;
}

// trim from end
static inline std::string &rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
    return s;
}

// trim from both ends
static inline std::string &trim(std::string &s) {
    return ltrim(rtrim(s));
}

NConfig::NConfig()
{

}

NConfig::~NConfig()
{

}

void NConfig::init(const std::string config_path)
{
    fstream file(config_path);
    if(!file.is_open()) cout << "Open File Failure..." << endl;
    string content;
    while (getline(file, content))
    {
        vector<string> sp = n_split(content, ':');
        if(sp[0] == "image_width")  image_width_ = stoi(sp[1]);
        if(sp[0] == "image_height") image_height_ = stoi(sp[1]);
        if(sp[0] == "dim")          dim_ = stoi(sp[1]);
        if(sp[0] == "train_file")   train_path_ = sp[1];
        if(sp[0] == "test_file")    test_path_ = sp[1];
        if(sp[0] == "save_file")    save_path_ = sp[1];
    }
}

vector<string> NConfig::n_split(std::string str, char sign)
{
    vector<string> sp;
    size_t start = 0, end = 0;

    while ((end = str.find(sign, start)) != (string::npos))
    {
        sp.push_back(str.substr(start, end - start));
        trim(sp.back());
        start = end + 1;
    }
    sp.push_back(str.substr(start));
    trim(sp.back());

    return  sp;
}