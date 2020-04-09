//
// Created by 许斌 on 2020/4/3.
//

#ifndef FACE_DETECTOR_N_CONFIG_H
#define FACE_DETECTOR_N_CONFIG_H

#include <string>

namespace n_config
{
class NConfig
{
private:

public:
    int image_width_;
    int image_height_;
    int dim_;

    std::string train_path_;
    std::string test_path_;
    std::string label_path_;
    std::string save_path_;

    NConfig();
    ~NConfig();

    void init(const std::string config_path);
    std::vector<std::string> n_split(std::string str, char sign);
};
}


#endif //FACE_DETECTOR_N_CONFIG_H
