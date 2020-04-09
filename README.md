# Face_Detector
```
.  
├── CMakeLists.txt  
├── README.md  
├── config                      // 配置文件
│   ├── config.txt              // pca参数配置
│   ├── label.txt               // 标签对应人脸
│   ├── test.txt                // 测试集路径及标签
│   └── train.txt               // 训练集路径及标签
├── creat_dataset.sh            // 创建配置文件脚本
├── data                        // 训练数据
│   └── data.txt                // 用于保存训练数据
├── demo
│   ├── demo.cpp                // 训练数据并识别
│   └── detector_demo.cpp       // 读取保存的训练数据并识别
├── include  
│   └── face_detector           
│       ├── face_detector.h     
│       └── n_config.h  
├── orl_faces                   // 数据集
├── src  
│   ├── face_detector.cpp       // pca
│   └── n_config.cpp            // 处理配置文件并读取参数
```
## 所需第三方库
```
- Opencv
- Eigen
```

## 运行步骤
数据集处理及编译：
```
- ./creat_dataset.sh        // 生成测试集和训练集的配置文件
- mkdir build
- cd build
- cmake ..
- make
```
运行：
```
- ./demo              // 训练数据并识别
- ./detector_demo     // 读取训练数据并识别
```