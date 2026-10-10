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
#ifndef SYSTEM_H
#define SYSTEM_H


#include <unistd.h>                 // Unix 标准函数定义
#include<stdio.h>                   // C 标准输入输出库
#include<stdlib.h>                  // C 通用工具库
#include<string>                    // C++ 字符串类
#include<thread>                    // C++11 线程库
#include<opencv2/core/core.hpp>     // opencv 模块

#include "Tracking.h"               // 跟踪
#include "FrameDrawer.h"            // 帧绘制
#include "MapDrawer.h"              // 地图绘制
#include "Atlas.h"                  // Atlas 多地图管理
#include "LocalMapping.h"           // 局部建图
#include "LoopClosing.h"            // 回环检测
#include "KeyFrameDatabase.h"       // 关键帧数据库
#include "ORBVocabulary.h"          // ORB 词袋
#include "Viewer.h"                 // 可视化
#include "ImuTypes.h"               // IMU 数据类型
#include "Settings.h"               // 系统参数设置


namespace ORB_SLAM3 // 命名空间
{

class Verbose // 日志
{
public:
    enum eLevel // 日志输出级别枚举
    {
        VERBOSITY_QUIET=0,          // 不输出
        VERBOSITY_NORMAL=1,         // 普通
        VERBOSITY_VERBOSE=2,        // 详细
        VERBOSITY_VERY_VERBOSE=3,   // 非常详细
        VERBOSITY_DEBUG=4           // 调试
    };

    static eLevel th; // 静态成员变量，类内定义，类外初始化

public:
    static void PrintMess(std::string str, eLevel lev)
    {
        if(lev <= th)
            cout << str << endl;
    }

    static void SetTh(eLevel _th)
    {
        th = _th;
    }
};

// 前向声明
class Viewer;
class FrameDrawer;
class MapDrawer;
class Atlas;
class Tracking;
class LocalMapping;
class LoopClosing;
class Settings;

class System // 系统类
{
public:
    // Input sensor
    enum eSensor{           // 传感器类型枚举
        MONOCULAR=0,        // 单目
        STEREO=1,           // 双目
        RGBD=2,             // RGB-D
        IMU_MONOCULAR=3,    // 单目 + IMU
        IMU_STEREO=4,       // 双目 + IMU
        IMU_RGBD=5,         // RGB-D + IMU
    };

    // File type
    enum FileType{      // 文件类型枚举
        TEXT_FILE=0,    // 文本
        BINARY_FILE=1,  // 二进制
    };

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW // Eigen 类型内存对齐宏
    // Initialize the SLAM system. It launches the Local Mapping, Loop Closing and Viewer threads.
    // SLAM 系统初始化（词袋文件路径，参数文件路径，传感器类型，是否使用可视化，初始帧数，序列名称），启动局部建图、回环检测和可视化线程
    System(const string &strVocFile, const string &strSettingsFile, const eSensor sensor, const bool bUseViewer = true, const int initFr = 0, const string &strSequence = std::string());

    // Proccess the given stereo frame. Images must be synchronized and rectified.
    // Input images: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
    // Returns the camera pose (empty if tracking fails).
    // 双目跟踪（左图，右图，时间戳，IMU 数据（默认为空），文件名（默认为空）），返回相机位姿（跟踪失败则为空）
    Sophus::SE3f TrackStereo(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timestamp, const vector<IMU::Point>& vImuMeas = vector<IMU::Point>(), string filename="");

    // Process the given rgbd frame. Depthmap must be registered to the RGB frame.
    // Input image: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
    // Input depthmap: Float (CV_32F).
    // Returns the camera pose (empty if tracking fails).
    // RGB-D 跟踪（图像，深度图，时间戳，IMU 数据（默认为空），文件名（默认为空）），返回相机位姿（跟踪失败则为空）
    Sophus::SE3f TrackRGBD(const cv::Mat &im, const cv::Mat &depthmap, const double &timestamp, const vector<IMU::Point>& vImuMeas = vector<IMU::Point>(), string filename="");

    // Proccess the given monocular frame and optionally imu data
    // Input images: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
    // Returns the camera pose (empty if tracking fails).
    // 单目跟踪（图像，时间戳，IMU 数据（默认为空），文件名（默认为空）），返回相机位姿（跟踪失败则为空）
    Sophus::SE3f TrackMonocular(const cv::Mat &im, const double &timestamp, const vector<IMU::Point>& vImuMeas = vector<IMU::Point>(), string filename="");


    // This stops local mapping thread (map building) and performs only camera tracking.
    void ActivateLocalizationMode(); // 停止局部建图线程（地图构建），仅执行相机跟踪。
    // This resumes local mapping thread and performs SLAM again.
    void DeactivateLocalizationMode(); // 恢复完整 SLAM

    // Returns true if there have been a big map change (loop closure, global BA)
    // since last call to this function
    bool MapChanged(); // 地图是否出现了大变化

    // Reset the system (clear Atlas or the active map)
    void Reset();           // 重置整个系统
    void ResetActiveMap();  // 重置当前活跃的地图

    // All threads will be requested to finish.
    // It waits until all threads have finished.
    // This function must be called before saving the trajectory.
    void Shutdown();    // 关闭系统，等待所有线程结束，必须在保存轨迹前调用
    bool isShutDown();  // 系统是否已经关闭

    // Save camera trajectory in the TUM RGB-D dataset format.
    // Only for stereo and RGB-D. This method does not work for monocular.
    // Call first Shutdown()
    // See format details at: http://vision.in.tum.de/data/datasets/rgbd-dataset
    void SaveTrajectoryTUM(const string &filename);

    // Save keyframe poses in the TUM RGB-D dataset format.
    // This method works for all sensor input.
    // Call first Shutdown()
    // See format details at: http://vision.in.tum.de/data/datasets/rgbd-dataset
    void SaveKeyFrameTrajectoryTUM(const string &filename);

    void SaveTrajectoryEuRoC(const string &filename);
    void SaveKeyFrameTrajectoryEuRoC(const string &filename);

    void SaveTrajectoryEuRoC(const string &filename, Map* pMap);
    void SaveKeyFrameTrajectoryEuRoC(const string &filename, Map* pMap);

    // Save data used for initialization debug
    void SaveDebugData(const int &iniIdx); // 保存用于初始化调试的数据

    // Save camera trajectory in the KITTI dataset format.
    // Only for stereo and RGB-D. This method does not work for monocular.
    // Call first Shutdown()
    // See format details at: http://www.cvlibs.net/datasets/kitti/eval_odometry.php
    void SaveTrajectoryKITTI(const string &filename);

    // TODO: Save/Load functions
    // SaveMap(const string &filename);
    // LoadMap(const string &filename);

    // Information from most recent processed frame
    // You can call this right after TrackMonocular (or stereo or RGBD)
    int GetTrackingState();                             // 返回最近处理帧的跟踪状态
    std::vector<MapPoint*> GetTrackedMapPoints();       // 返回当前帧跟踪到的地图点
    std::vector<cv::KeyPoint> GetTrackedKeyPointsUn();  // 返回当前帧跟踪到的去畸变关键点

    // For debugging
    double GetTimeFromIMUInit();  // 获取从 IMU 初始化开始的时间
    bool isLost();                // 判断跟踪是否丢失
    bool isFinished();            // 判断系统是否完成

    void ChangeDataset(); // 切换序列

    float GetImageScale(); // 获取图像缩放比例

#ifdef REGISTER_TIMES // 记录时间（REGISTER_TIMES 宏定义打开时，位于 include/Setting.h 中）
    void InsertRectTime(double& time);
    void InsertResizeTime(double& time);
    void InsertTrackTime(double& time);
#endif

private:

    void SaveAtlas(int type); // 保存地图集
    bool LoadAtlas(int type); // 加载地图集

    string CalculateCheckSum(string filename, int type); // 计算文件校验和，用于验证地图文件

    // Input sensor
    eSensor mSensor; // 传感器类型

    // ORB vocabulary used for place recognition and feature matching.
    ORBVocabulary* mpVocabulary; // 指向 ORB 词袋的指针

    // KeyFrame database for place recognition (relocalization and loop detection).
    KeyFrameDatabase* mpKeyFrameDatabase; // 指向关键帧数据库的指针

    // Map structure that stores the pointers to all KeyFrames and MapPoints.
    //Map* mpMap;
    Atlas* mpAtlas; // 指向地图集的指针

    // Tracker. It receives a frame and computes the associated camera pose.
    // It also decides when to insert a new keyframe, create some new MapPoints and
    // performs relocalization if tracking fails.
    Tracking* mpTracker; // 指向跟踪器的指针

    // Local Mapper. It manages the local map and performs local bundle adjustment.
    LocalMapping* mpLocalMapper; // 指向局部建图器的指针

    // Loop Closer. It searches loops with every new keyframe. If there is a loop it performs
    // a pose graph optimization and full bundle adjustment (in a new thread) afterwards.
    LoopClosing* mpLoopCloser; // 指向回环检测器的指针

    // The viewer draws the map and the current camera pose. It uses Pangolin.
    Viewer* mpViewer; // 指向可视化器的指针

    FrameDrawer* mpFrameDrawer; // 指向帧绘制器的指针
    MapDrawer* mpMapDrawer;     // 指向地图绘制器的指针

    // System threads: Local Mapping, Loop Closing, Viewer.
    // The Tracking thread "lives" in the main execution thread that creates the System object.
    std::thread* mptLocalMapping; // 指向局部建图线程的指针
    std::thread* mptLoopClosing;  // 指向回环检测线程的指针
    std::thread* mptViewer;       // 指向可视化线程的指针

    // Reset flag
    std::mutex mMutexReset;   // 保护重置标志的互斥锁
    bool mbReset;             // 重置整个系统标志
    bool mbResetActiveMap;    // 重置当前活跃的地图标志

    // Change mode flags
    std::mutex mMutexMode;              // 保护模式切换标志的互斥锁
    bool mbActivateLocalizationMode;    // 激活定位模式标志（停止局部建图线程，仅执行相机跟踪）
    bool mbDeactivateLocalizationMode;  // 停用定位模式标志（恢复完整 SLAM）

    // Shutdown flag
    bool mbShutDown; // 关闭系统标志

    // Tracking state
    int mTrackingState;                             // 跟踪状态
    std::vector<MapPoint*> mTrackedMapPoints;       // 当前跟踪到的地图点
    std::vector<cv::KeyPoint> mTrackedKeyPointsUn;  // 当前跟踪到的去畸变关键点
    std::mutex mMutexState;                         // 保护上述状态数据的互斥锁

    //
    string mStrLoadAtlasFromFile;   // 地图集文件加载路径
    string mStrSaveAtlasToFile;     // 地图集文件保存路径

    string mStrVocabularyFilePath;  // 词袋文件路径

    Settings* settings_;            // 指向 setting 的指针
};

}// namespace ORB_SLAM

// 结束头文件保护
#endif // SYSTEM_H
