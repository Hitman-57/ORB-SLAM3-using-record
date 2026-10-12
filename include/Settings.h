/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

// 头文件保护宏
#ifndef ORB_SLAM3_SETTINGS_H
#define ORB_SLAM3_SETTINGS_H


// Flag to activate the measurement of time in each process (track,localmap, place recognition).
//#define REGISTER_TIMES

#include "CameraModels/GeometricCamera.h" // 几何相机模型基类

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>

namespace ORB_SLAM3 {

    class System; // 前向声明

    //TODO: change to double instead of float

    // Settings 类定义
    class Settings {
    public:
        /*
         * Enum for the different camera types implemented
         */
        enum CameraType {       // 相机类型枚举
            PinHole = 0,        // 针孔相机
            Rectified = 1,      // 已校正图像
            KannalaBrandt = 2   // 鱼眼相机
        };

        /*
         * Delete default constructor
         */
        Settings() = delete; // 删除默认构造函数

        /*
         * Constructor from file
         */
        Settings(const std::string &configFile, const int& sensor); // 构造函数声明

        /*
         * Ostream operator overloading to dump settings to the terminal
         */
        friend std::ostream &operator<<(std::ostream &output, const Settings &s); // 友元函数 operator<< 声明

        /*
         * Getter methods
         */
        CameraType cameraType() {return cameraType_;}       // 相机类型
        GeometricCamera* camera1() {return calibration1_;}  // 相机 1 标定参数
        GeometricCamera* camera2() {return calibration2_;}  // 相机 2 标定参数
        // 相机 1 畸变参数
        cv::Mat camera1DistortionCoef() {return cv::Mat(vPinHoleDistorsion1_.size(),1,CV_32F,vPinHoleDistorsion1_.data());}
        // 相机 2 畸变参数
        cv::Mat camera2DistortionCoef() {return cv::Mat(vPinHoleDistorsion2_.size(),1,CV_32F,vPinHoleDistorsion1_.data());}

        // 双目
        Sophus::SE3f Tlr() {return Tlr_;}   // 双目外参
        float bf() {return bf_;}            // 基线乘以焦距
        float b() {return b_;}              // 基线
        float thDepth() {return thDepth_;}  // 深度阈值

        bool needToUndistort() {return bNeedToUndistort_;} // 是否需要去畸变

        cv::Size newImSize() {return newImSize_;}          // 缩放后图像尺寸
        float fps() {return fps_;}                         // 帧率
        bool rgb() {return bRGB_;}                         // 是否 RGB
        bool needToResize() {return bNeedToResize1_;}      // 是否需要缩放
        bool needToRectify() {return bNeedToRectify_;}     // 是否需要校正（双目）

        float noiseGyro() {return noiseGyro_;}                  // 陀螺仪噪声
        float noiseAcc() {return noiseAcc_;}                    // 加速度计噪声
        float gyroWalk() {return gyroWalk_;}                    // 陀螺仪随机游走噪声
        float accWalk() {return accWalk_;}                      // 加速度随机游走计噪声
        float imuFrequency() {return imuFrequency_;}            // IMU 采样率
        Sophus::SE3f Tbc() {return Tbc_;}                       // 相机相对于 IMU 的位姿
        bool insertKFsWhenLost() {return insertKFsWhenLost_;}   // 丢失是是否插入关键帧

        float depthMapFactor() {return depthMapFactor_;} // 深度图因子（RGB-D）

        int nFeatures() {return nFeatures_;}        // ORB 特征点数量
        int nLevels() {return nLevels_;}            // 金字塔层数
        float initThFAST() {return initThFAST_;}    // 初始 FAST
        float minThFAST() {return minThFAST_;}      // 最小 FAST
        float scaleFactor() {return scaleFactor_;}  // 尺度因子

        //  Viewer 参数
        float keyFrameSize() {return keyFrameSize_;}
        float keyFrameLineWidth() {return keyFrameLineWidth_;}
        float graphLineWidth() {return graphLineWidth_;}
        float pointSize() {return pointSize_;}
        float cameraSize() {return cameraSize_;}
        float cameraLineWidth() {return cameraLineWidth_;}
        float viewPointX() {return viewPointX_;}
        float viewPointY() {return viewPointY_;}
        float viewPointZ() {return viewPointZ_;}
        float viewPointF() {return viewPointF_;}
        float imageViewerScale() {return imageViewerScale_;}

        // Altas 加载和保存地址
        std::string atlasLoadFile() {return sLoadFrom_;}
        std::string atlasSaveFile() {return sSaveto_;}

        float thFarPoints() {return thFarPoints_;} // 远点阈值

        // 校正映射
        cv::Mat M1l() {return M1l_;}
        cv::Mat M2l() {return M2l_;}
        cv::Mat M1r() {return M1r_;}
        cv::Mat M2r() {return M2r_;}

    private:
        template<typename T> // 参数读取函数模板
        T readParameter(cv::FileStorage& fSettings, const std::string& name, bool& found,const bool required = true){
            cv::FileNode node = fSettings[name]; // 从 fSettings 中取 name 节点
            if(node.empty()){ // 参数为空
                if(required){ // 需要该参数
                    std::cerr << name << " required parameter does not exist, aborting..." << std::endl;
                    exit(-1);
                }
                else{ // 不需要该参数
                    std::cerr << name << " optional parameter does not exist..." << std::endl;
                    found = false;
                    return T();
                }

            }
            else{ // 参数不为空
                found = true;
                return (T) node;
            }
        }

        // 参数读取函数
        void readCamera1(cv::FileStorage& fSettings);
        void readCamera2(cv::FileStorage& fSettings);
        void readImageInfo(cv::FileStorage& fSettings);
        void readIMU(cv::FileStorage& fSettings);
        void readRGBD(cv::FileStorage& fSettings);
        void readORB(cv::FileStorage& fSettings);
        void readViewer(cv::FileStorage& fSettings);
        void readLoadAndSave(cv::FileStorage& fSettings);
        void readOtherParameters(cv::FileStorage& fSettings);

        void precomputeRectificationMaps();

        int sensor_;                // 传感器类型 
        CameraType cameraType_;     // 相机类型

        /*
         * Visual stuff
         */
        GeometricCamera* calibration1_, *calibration2_;                 // 当前相机内参
        GeometricCamera* originalCalib1_, *originalCalib2_;             // 原始相机内参
        std::vector<float> vPinHoleDistorsion1_, vPinHoleDistorsion2_;  // 针孔相机畸变系数（鱼眼相机畸系数变一并存入相机内参）

        cv::Size originalImSize_, newImSize_; // 原始图像大小，当前图像大小 
        float fps_;                           // 帧率
        bool bRGB_;                           // 是否 RGB

        bool bNeedToUndistort_;                 // 是否去畸变
        bool bNeedToRectify_;                   // 是否校正（双目）
        bool bNeedToResize1_, bNeedToResize2_;  // 是否 risize

        Sophus::SE3f Tlr_; // 双目外参
        float thDepth_;    // 深度阈值
        float bf_, b_;     // 基线*焦距，基线
 
        /*
         * 双目较正映射矩阵 Rectification stuff
         */
        cv::Mat M1l_, M2l_;
        cv::Mat M1r_, M2r_;

        /*
         * Inertial stuff
         */
        float noiseGyro_, noiseAcc_; // IMU 噪声参数
        float gyroWalk_, accWalk_;   // IMU 游走噪声参数
        float imuFrequency_;         // IMU 采样率
        Sophus::SE3f Tbc_;           // 相机相对于 IMU 的位姿
        bool insertKFsWhenLost_;     // 丢失时是否插入关键帧

        /*
         * RGBD stuff
         */
        float depthMapFactor_; // 深度图因子（RGB-D）

        /*
         * ORB stuff
         */
        int nFeatures_;               // ORB 特征点数 
        float scaleFactor_;           // 缩放因子
        int nLevels_;                 // 金字塔层数
        int initThFAST_, minThFAST_;  // 初始 FAST，最小 FAST

        /*
         * 可视化参数 Viewer stuff
         */
        float keyFrameSize_;
        float keyFrameLineWidth_;
        float graphLineWidth_;
        float pointSize_;
        float cameraSize_;
        float cameraLineWidth_;
        float viewPointX_, viewPointY_, viewPointZ_, viewPointF_;
        float imageViewerScale_;

        /*
         * Atlas 地图加载和保存路径 Save & load maps
         */
        std::string sLoadFrom_, sSaveto_;

        /*
         * 远点阈值 Other stuff
         */
        float thFarPoints_;

    };
};


#endif //ORB_SLAM3_SETTINGS_H
