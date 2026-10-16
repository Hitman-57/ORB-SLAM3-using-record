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
#ifndef TRACKING_H
#define TRACKING_H

#include <opencv2/core/core.hpp>                // opencv 核心模块
#include <opencv2/features2d/features2d.hpp>    // opencv 特征点模块

#include "Viewer.h"             // 可视化
#include "FrameDrawer.h"        // 帧绘制器
#include "Atlas.h"              // Atlas 地图集
#include "LocalMapping.h"       // 局部建图线程
#include "LoopClosing.h"        // 回环检测线程
#include "Frame.h"              // 帧类
#include "ORBVocabulary.h"      // ORB 词袋模型
#include "KeyFrameDatabase.h"   // 关键帧数据库
#include "ORBextractor.h"       // ORB 特征提取
#include "MapDrawer.h"          // 地图绘制器
#include "System.h"             // 主系统
#include "ImuTypes.h"           // IMU 数据类型
#include "Settings.h"           // 配置参数

#include "GeometricCamera.h"    // 几何相机模型

#include <mutex>                // 互斥锁
#include <unordered_set>        // 哈希集合

namespace ORB_SLAM3
{

// 前向声明
class Viewer;
class FrameDrawer;
class Atlas;
class LocalMapping;
class LoopClosing;
class System;
class Settings;

// Tracking 类定义
class Tracking
{  

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW // Eigen 类型内存对齐宏
    // 构造函数（系统，词袋，帧绘制器，地图绘制器，Atlas 地图集，帧数据库，参数文件路径，传感器类型，配置参数，序列名（默认为空））
    Tracking(System* pSys, ORBVocabulary* pVoc, FrameDrawer* pFrameDrawer, MapDrawer* pMapDrawer, Atlas* pAtlas,
             KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor, Settings* settings, const string &_nameSeq=std::string());
    // 析构函数，释放资源
    ~Tracking();

    // 解析配置参数文件 Parse the config file
    bool ParseCamParamFile(cv::FileStorage &fSettings); // 相机相关参数
    bool ParseORBParamFile(cv::FileStorage &fSettings); // ORB 特征提取相关参数
    bool ParseIMUParamFile(cv::FileStorage &fSettings); // IMU 相关参数

    // Preprocess the input and call Track(). Extract features and performs stereo matching.
    // 图像抓取、预处理和 Track() 调用
    Sophus::SE3f GrabImageStereo(const cv::Mat &imRectLeft,const cv::Mat &imRectRight, const double &timestamp, string filename);
    Sophus::SE3f GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp, string filename);
    Sophus::SE3f GrabImageMonocular(const cv::Mat &im, const double &timestamp, string filename);
    // 抓取帧间 IMU 数据
    void GrabImuData(const IMU::Point &imuMeasurement);
    
    void SetLocalMapper(LocalMapping* pLocalMapper); // 设置指向局部建图线程的指针
    void SetLoopClosing(LoopClosing* pLoopClosing);  // 设置指向回环检测线程的指针
    void SetViewer(Viewer* pViewer);                 // 设置指向可视化线程的指针
    void SetStepByStep(bool bSet); // 设置单步调试模式
    bool GetStepByStep();          // 获取单步调试模式状态

    // Load new settings
    // The focal lenght should be similar or scale prediction will fail when projecting points
    void ChangeCalibration(const string &strSettingPath); // 更改相机标定？？？

    // Use this function if you have deactivated local mapping and you only want to localize the camera.
    void InformOnlyTracking(const bool &flag); // 仅定位
    
    // 用 IMU 数据更新当前帧
    void UpdateFrameIMU(const float s, const IMU::Bias &b, KeyFrame* pCurrentKeyFrame);
    KeyFrame* GetLastKeyFrame() // 返回最后一个关键帧
    {
        return mpLastKeyFrame;
    }

    void CreateMapInAtlas(); // 在 Atlas 中创建地图
    //std::mutex mMutexTracks;

    //--
    void NewDataset();       // 新建数据集
    int GetNumberDataset();  // 返回当前数据集编号
    int GetMatchesInliers(); // 返回当前帧匹配点数量

    //DEBUG
    void SaveSubTrajectory(string strNameFile_frames, string strNameFile_kf, string strFolder="");
    void SaveSubTrajectory(string strNameFile_frames, string strNameFile_kf, Map* pMap);

    float GetImageScale(); // 返回图像缩放系数

#ifdef REGISTER_LOOP // 若定义了 REGISTER_LOOP，则提供停止、释放接口
    void RequestStop();
    bool isStopped();
    void Release();
    bool stopRequested();
#endif

public:

    // Tracking states
    enum eTrackingState{     // 跟踪状态枚举
        SYSTEM_NOT_READY=-1, // 未准备
        NO_IMAGES_YET=0,     // 无图像
        NOT_INITIALIZED=1,   // 未初始化
        OK=2,                // OK
        RECENTLY_LOST=3,     // 丢失（最近）
        LOST=4,              // 完全丢失
        OK_KLT=5             // OK 但使用光流跟踪
    };

    eTrackingState mState;              // 当前跟踪状态
    eTrackingState mLastProcessedState; // 上一次处理后的状态

    // Input sensor
    int mSensor; // 传感器类型

    // Current Frame
    Frame mCurrentFrame; // 当前帧
    Frame mLastFrame;    // 上一帧

    cv::Mat mImGray; // 灰度图

    // 单目相关变量初始化 Initialization Variables (Monocular)
    std::vector<int> mvIniLastMatches;       // 初始化过程中上一帧的匹配索引
    std::vector<int> mvIniMatches;           // 初始帧与当前帧的匹配索引
    std::vector<cv::Point2f> mvbPrevMatched; // 上一帧匹配的特征点坐标
    std::vector<cv::Point3f> mvIniP3D;       // 初始化三角化的三维点
    Frame mInitialFrame;                     // 单目初始帧

    // Lists used to recover the full camera trajectory at the end of the execution.
    // Basically we store the reference keyframe for each frame and its relative transformation
    list<Sophus::SE3f> mlRelativeFramePoses; // 参考关键帧相对于每帧的位姿
    list<KeyFrame*> mlpReferences;           // 每帧对应的参考关键帧
    list<double> mlFrameTimes;               // 帧时间戳
    list<bool> mlbLost;                      // 该帧是否丢失

    // frames with estimated pose
    int mTrackedFr; // 成功进入局部地图跟踪的帧数
    bool mbStep;    // 是否单步调试

    // True if local mapping is deactivated and we are performing only localization
    bool mbOnlyTracking; // 是否仅定位

    void Reset(bool bLocMap = false);           // 重置系统
    void ResetActiveMap(bool bLocMap = false);  // 重置当前活跃地图

    float mMeanTrack;       // 平均跟踪时间
    bool mbInitWith3KFs;    // 是否使用前三个关键帧进行初始化
    double t0;              // time-stamp of first read frame
    double t0vis;           // time-stamp of first inserted keyframe
    double t0IMU;           // time-stamp of IMU initialization
    bool mFastInit = false; // 快速初始化标志


    vector<MapPoint*> GetLocalMapMPS(); // 返回当前局部地图点列表

    bool mbWriteStats; // 是否写统计信息

#ifdef REGISTER_TIMES // 时间统计（REGISTER_TIMES 宏定义打开时，位于 include/Setting.h 中）
    void LocalMapStats2File();
    void TrackStats2File();
    void PrintTimeStats();

    vector<double> vdRectStereo_ms;   // 双目校正
    vector<double> vdResizeImage_ms;  // 图像缩放
    vector<double> vdORBExtract_ms;   // 特征提取
    vector<double> vdStereoMatch_ms;  // 双目匹配
    vector<double> vdIMUInteg_ms;     // IMU 积分
    vector<double> vdPosePred_ms;     // 位姿预测
    vector<double> vdLMTrack_ms;      // 局部地图跟踪
    vector<double> vdNewKF_ms;        // 新关键帧决策
    vector<double> vdTrackTotal_ms;   // 总跟踪时间
#endif

protected:

    // Main tracking function. It is independent of the input sensor.
    void Track(); // 跟踪线程主函数

    // Map initialization for stereo and RGB-D
    void StereoInitialization(); // 双目或 RGB-D 地图初始化

    // Map initialization for monocular
    void MonocularInitialization(); // 单目地图初始化
    //void CreateNewMapPoints();
    void CreateInitialMapMonocular(); // 创建单目初始地图

    void CheckReplacedInLastFrame(); // 检查上一帧中是否有地图点被替换
    bool TrackReferenceKeyFrame();   // 基于参考关键帧做位姿估计
    void UpdateLastFrame();          // 更新上一帧信息（位姿，临时地图点）
    bool TrackWithMotionModel();     // 恒速运动模型做位姿估计
    bool PredictStateIMU();          // 使用 IMU 预积分做位姿估计

    bool Relocalization(); // 重定位

    void UpdateLocalMap();       // 更新局部地图
    void UpdateLocalPoints();    // 更新局部地图点
    void UpdateLocalKeyFrames(); // 更新局部关键帧

    bool TrackLocalMap();     // 局部地图跟踪
    void SearchLocalPoints(); // 在局部地图点中搜索当前帧匹配

    bool NeedNewKeyFrame();   // 判断是否需要插入新关键帧
    void CreateNewKeyFrame(); // 创建并插入新关键帧

    // Perform preintegration from last frame
    void PreintegrateIMU(); // 帧间 IMU 数据预积分

    // Reset IMU biases and compute frame velocity
    void ResetFrameIMU(); // 帧 IMU 重置（空实现）

    bool mbMapUpdated; // 地图是否在本帧更新

    // Imu preintegration from last frame
    IMU::Preintegrated *mpImuPreintegratedFromLastKF; // 上一关键帧到当前帧的 IMU 预积分对象

    // Queue of IMU measurements between frames
    std::list<IMU::Point> mlQueueImuData; // IMU 数据队列

    // Vector of IMU measurements from previous to current frame (to be filled by PreintegrateIMU)
    std::vector<IMU::Point> mvImuFromLastFrame; // 帧间 IMU 数据
    std::mutex mMutexImuQueue;  // IMU 队列保护锁

    // Imu calibration parameters
    IMU::Calib *mpImuCalib; // IMU 标定参数

    // Last Bias Estimation (at keyframe creation)
    IMU::Bias mLastBias; // 最后一次估计的 IMU 偏置（关键帧创建时更新）

    // In case of performing only localization, this flag is true when there are no matches to
    // points in the map. Still tracking will continue if there are enough matches with temporal points.
    // In that case we are doing visual odometry. The system will try to do relocalization to recover
    // "zero-drift" localization to the map.
    bool mbVO; // 是否处于视觉里程计模式

    //Other Thread Pointers
    LocalMapping* mpLocalMapper; // 局部建图线程指针
    LoopClosing* mpLoopClosing;  // 回环检测线程指针

    //ORB
    ORBextractor* mpORBextractorLeft, *mpORBextractorRight; // 左图特征提取器，右图特征提取器
    ORBextractor* mpIniORBextractor; // 初始化阶段的特征提取器

    //BoW
    ORBVocabulary* mpORBVocabulary; // ORB 词袋
    KeyFrameDatabase* mpKeyFrameDB; // 关键帧数据库

    // 单目初始化标志 Initalization (only for monocular)
    bool mbReadyToInitializate; // 是否准备好初始化
    bool mbSetInit;             // 是否已设置初始化

    //Local Map
    KeyFrame* mpReferenceKF;                  // 当前参考关键帧
    std::vector<KeyFrame*> mvpLocalKeyFrames; // 局部关键帧集合
    std::vector<MapPoint*> mvpLocalMapPoints; // 局部地图点集合
    
    // System
    System* mpSystem; // 系统对象指针
    
    //Drawers
    Viewer* mpViewer;           // 可视化线程
    FrameDrawer* mpFrameDrawer; // 帧绘制器
    MapDrawer* mpMapDrawer;     // 地图绘制器
    bool bStepByStep;           // 是否单步调试

    //Atlas
    Atlas* mpAtlas; // Atlas 地图集

    //Calibration matrix
    cv::Mat mK;          // opencv 格式内参矩阵
    Eigen::Matrix3f mK_; // eigen 格式内参矩阵
    cv::Mat mDistCoef;   // opencv 格式畸变系数矩阵
    float mbf;           // 基线*焦距
    float mImageScale;   // 图像缩放系数

    float mImuFreq;      // IMU 采样率
    double mImuPer;      // IMU 周期（采样时间间隔？？？）
    bool mInsertKFsLost; // 丢失时是否插入关键帧

    // 新关键帧插入规则（根据帧率） New KeyFrame rules (according to fps)
    int mMinFrames; // 最小帧数
    int mMaxFrames; // 最大帧数

    int mnFirstImuFrameId;  // 第一帧 IMU 的帧 ID
    int mnFramesToResetIMU; // 重置 IMU 的帧数

    // Threshold close/far points
    // Points seen as close by the stereo/RGBD sensor are considered reliable
    // and inserted from just one frame. Far points requiere a match in two keyframes.
    float mThDepth; // 近/远点阈值

    // For RGB-D inputs only. For some datasets (e.g. TUM) the depthmap values are scaled.
    float mDepthMapFactor; // 深度图因子（RGB-D）

    //Current matches in frame
    int mnMatchesInliers; // 特征点匹配数量

    //Last Frame, KeyFrame and Relocalisation Info
    KeyFrame* mpLastKeyFrame;        // 最后的关键帧
    unsigned int mnLastKeyFrameId;   // 最后的关键帧 ID
    unsigned int mnLastRelocFrameId; // 最后的重定位帧 ID
    double mTimeStampLost;           // 丢失时的时间戳
    double time_recently_lost;       // 最近丢失持续时间

    unsigned int mnFirstFrameId;    // 首帧 ID
    unsigned int mnInitialFrameId;  // 初始化帧 ID
    unsigned int mnLastInitFrameId; // 最后初始化帧 ID

    bool mbCreatedMap; // 是否已创建地图

    //Motion Model
    bool mbVelocity{false}; // 是否已有速度估计
    Sophus::SE3f mVelocity; // 恒速模型的速度

    //Color order (true RGB, false BGR, ignored if grayscale)
    bool mbRGB; // 颜色顺序（灰度图忽略）

    list<MapPoint*> mlpTemporalPoints; // 临时地图点列表

    //int nMapChangeIndex;

    int mnNumDataset; // 当前序列比编号

    ofstream f_track_stats; // 跟踪统计
    ofstream f_track_times; // 跟踪耗时

    double mTime_PreIntIMU;     // IMU 预积分耗时
    double mTime_PosePred;      // 位姿预测耗时
    double mTime_LocalMapTrack; // 局部地图跟踪耗时
    double mTime_NewKF_Dec;     // 新关键帧决策耗时

    GeometricCamera* mpCamera, *mpCamera2; // 几何相机模型

    int initID, lastID; // 初始与最后 ID

    Sophus::SE3f mTlr; // 双目相机外参

    void newParameterLoader(Settings* settings); // 配置参数加载

#ifdef REGISTER_LOOP
    bool Stop();

    bool mbStopped;
    bool mbStopRequested;
    bool mbNotStop;
    std::mutex mMutexStop;
#endif

public:
    cv::Mat mImRight; // 右图像（双目）
};

} //namespace ORB_SLAM

#endif // TRACKING_H
