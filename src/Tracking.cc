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


#include "Tracking.h"

#include "ORBmatcher.h"     // ORB 特征匹配
#include "FrameDrawer.h"    // 帧绘制器
#include "Converter.h"      // 转换库
#include "G2oTypes.h"       // g2o 优化类型
#include "Optimizer.h"      // 位姿优化器
#include "Pinhole.h"        // 针孔相机模型
#include "KannalaBrandt8.h" // 鱼眼相机模型
#include "MLPnPsolver.h"    // MLPnP
#include "GeometricTools.h" // 几何工具

#include <iostream> // C++ 标准输入输出

#include <mutex>  // 互斥锁
#include <chrono> // 计时


using namespace std;

namespace ORB_SLAM3
{

// 构造函数定义（系统，词袋，帧绘制器，地图绘制器，Atlas 地图集，帧数据库，参数文件路径，传感器类型，配置参数，序列名（默认为空））
Tracking::Tracking(System *pSys, ORBVocabulary* pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Atlas *pAtlas, 
    KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor, Settings* settings, const string &_nameSeq):
    mState(NO_IMAGES_YET), mSensor(sensor), mTrackedFr(0), mbStep(false),
    mbOnlyTracking(false), mbMapUpdated(false), mbVO(false), mpORBVocabulary(pVoc), mpKeyFrameDB(pKFDB),
    mbReadyToInitializate(false), mpSystem(pSys), mpViewer(NULL), bStepByStep(false),
    mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpAtlas(pAtlas), mnLastRelocFrameId(0), time_recently_lost(5.0),
    mnInitialFrameId(0), mbCreatedMap(false), mnFirstFrameId(0), mpCamera2(nullptr), mpLastKeyFrame(static_cast<KeyFrame*>(NULL))
{
    // Load camera parameters from settings file
    if(settings){ // 实际就是 strSettingPath 的配置文件解析出来的参数
        newParameterLoader(settings);
    }
    else{ // 从 strSettingPath 的配置文件中解析（通常都会传入解析好的参数，该部分意义不明）
        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

        bool b_parse_cam = ParseCamParamFile(fSettings);
        if(!b_parse_cam)
        {
            std::cout << "*Error with the camera parameters in the config file*" << std::endl;
        }

        // Load ORB parameters
        bool b_parse_orb = ParseORBParamFile(fSettings);
        if(!b_parse_orb)
        {
            std::cout << "*Error with the ORB parameters in the config file*" << std::endl;
        }

        bool b_parse_imu = true;
        if(sensor==System::IMU_MONOCULAR || sensor==System::IMU_STEREO || sensor==System::IMU_RGBD)
        {
            b_parse_imu = ParseIMUParamFile(fSettings);
            if(!b_parse_imu)
            {
                std::cout << "*Error with the IMU parameters in the config file*" << std::endl;
            }

            mnFramesToResetIMU = mMaxFrames;
        }

        if(!b_parse_cam || !b_parse_orb || !b_parse_imu)
        {
            std::cerr << "**ERROR in the config file, the format is not correct**" << std::endl;
            try
            {
                throw -1;
            }
            catch(exception &e)
            {

            }
        }
    }

    initID = 0; lastID = 0; // 初始/最后帧 ID，置 0
    mbInitWith3KFs = false; // 是否用三个关键帧初始化，置 false
    mnNumDataset = 0;       // 数据集编号，置 0

    // 打印相机数量、ID 和类型
    vector<GeometricCamera*> vpCams = mpAtlas->GetAllCameras();
    std::cout << "There are " << vpCams.size() << " cameras in the atlas" << std::endl;
    for(GeometricCamera* pCam : vpCams)
    {
        std::cout << "Camera " << pCam->GetId();
        if(pCam->GetType() == GeometricCamera::CAM_PINHOLE)
        {
            std::cout << " is pinhole" << std::endl;
        }
        else if(pCam->GetType() == GeometricCamera::CAM_FISHEYE)
        {
            std::cout << " is fisheye" << std::endl;
        }
        else
        {
            std::cout << " is unknown" << std::endl;
        }
    }

#ifdef REGISTER_TIMES // 时间统计清空（REGISTER_TIMES 宏定义打开时，位于 include/Setting.h 中）
    vdRectStereo_ms.clear();
    vdResizeImage_ms.clear();
    vdORBExtract_ms.clear();
    vdStereoMatch_ms.clear();
    vdIMUInteg_ms.clear();
    vdPosePred_ms.clear();
    vdLMTrack_ms.clear();
    vdNewKF_ms.clear();
    vdTrackTotal_ms.clear();
#endif
}

#ifdef REGISTER_TIMES // 时间统计（REGISTER_TIMES 宏定义打开时，位于 include/Setting.h 中）
double calcAverage(vector<double> v_times)
{
    double accum = 0;
    for(double value : v_times)
    {
        accum += value;
    }

    return accum / v_times.size();
}

double calcDeviation(vector<double> v_times, double average)
{
    double accum = 0;
    for(double value : v_times)
    {
        accum += pow(value - average, 2);
    }
    return sqrt(accum / v_times.size());
}

double calcAverage(vector<int> v_values)
{
    double accum = 0;
    int total = 0;
    for(double value : v_values)
    {
        if(value == 0)
            continue;
        accum += value;
        total++;
    }

    return accum / total;
}

double calcDeviation(vector<int> v_values, double average)
{
    double accum = 0;
    int total = 0;
    for(double value : v_values)
    {
        if(value == 0)
            continue;
        accum += pow(value - average, 2);
        total++;
    }
    return sqrt(accum / total);
}

void Tracking::LocalMapStats2File()
{
    ofstream f;
    f.open("LocalMapTimeStats.txt");
    f << fixed << setprecision(6);
    f << "#Stereo rect[ms], MP culling[ms], MP creation[ms], LBA[ms], KF culling[ms], Total[ms]" << endl;
    for(int i=0; i<mpLocalMapper->vdLMTotal_ms.size(); ++i)
    {
        f << mpLocalMapper->vdKFInsert_ms[i] << "," << mpLocalMapper->vdMPCulling_ms[i] << ","
          << mpLocalMapper->vdMPCreation_ms[i] << "," << mpLocalMapper->vdLBASync_ms[i] << ","
          << mpLocalMapper->vdKFCullingSync_ms[i] <<  "," << mpLocalMapper->vdLMTotal_ms[i] << endl;
    }

    f.close();

    f.open("LBA_Stats.txt");
    f << fixed << setprecision(6);
    f << "#LBA time[ms], KF opt[#], KF fixed[#], MP[#], Edges[#]" << endl;
    for(int i=0; i<mpLocalMapper->vdLBASync_ms.size(); ++i)
    {
        f << mpLocalMapper->vdLBASync_ms[i] << "," << mpLocalMapper->vnLBA_KFopt[i] << ","
          << mpLocalMapper->vnLBA_KFfixed[i] << "," << mpLocalMapper->vnLBA_MPs[i] << ","
          << mpLocalMapper->vnLBA_edges[i] << endl;
    }


    f.close();
}

void Tracking::TrackStats2File()
{
    ofstream f;
    f.open("SessionInfo.txt");
    f << fixed;
    f << "Number of KFs: " << mpAtlas->GetAllKeyFrames().size() << endl;
    f << "Number of MPs: " << mpAtlas->GetAllMapPoints().size() << endl;

    f << "OpenCV version: " << CV_VERSION << endl;

    f.close();

    f.open("TrackingTimeStats.txt");
    f << fixed << setprecision(6);

    f << "#Image Rect[ms], Image Resize[ms], ORB ext[ms], Stereo match[ms], IMU preint[ms], Pose pred[ms], LM track[ms], KF dec[ms], Total[ms]" << endl;

    for(int i=0; i<vdTrackTotal_ms.size(); ++i)
    {
        double stereo_rect = 0.0;
        if(!vdRectStereo_ms.empty())
        {
            stereo_rect = vdRectStereo_ms[i];
        }

        double resize_image = 0.0;
        if(!vdResizeImage_ms.empty())
        {
            resize_image = vdResizeImage_ms[i];
        }

        double stereo_match = 0.0;
        if(!vdStereoMatch_ms.empty())
        {
            stereo_match = vdStereoMatch_ms[i];
        }

        double imu_preint = 0.0;
        if(!vdIMUInteg_ms.empty())
        {
            imu_preint = vdIMUInteg_ms[i];
        }

        f << stereo_rect << "," << resize_image << "," << vdORBExtract_ms[i] << "," << stereo_match << "," << imu_preint << ","
          << vdPosePred_ms[i] <<  "," << vdLMTrack_ms[i] << "," << vdNewKF_ms[i] << "," << vdTrackTotal_ms[i] << endl;
    }

    f.close();
}

void Tracking::PrintTimeStats()
{
    // Save data in files
    TrackStats2File();
    LocalMapStats2File();


    ofstream f;
    f.open("ExecMean.txt");
    f << fixed;
    //Report the mean and std of each one
    std::cout << std::endl << " TIME STATS in ms (mean$\\pm$std)" << std::endl;
    f << " TIME STATS in ms (mean$\\pm$std)" << std::endl;
    cout << "OpenCV version: " << CV_VERSION << endl;
    f << "OpenCV version: " << CV_VERSION << endl;
    std::cout << "---------------------------" << std::endl;
    std::cout << "Tracking" << std::setprecision(5) << std::endl << std::endl;
    f << "---------------------------" << std::endl;
    f << "Tracking" << std::setprecision(5) << std::endl << std::endl;
    double average, deviation;
    if(!vdRectStereo_ms.empty())
    {
        average = calcAverage(vdRectStereo_ms);
        deviation = calcDeviation(vdRectStereo_ms, average);
        std::cout << "Stereo Rectification: " << average << "$\\pm$" << deviation << std::endl;
        f << "Stereo Rectification: " << average << "$\\pm$" << deviation << std::endl;
    }

    if(!vdResizeImage_ms.empty())
    {
        average = calcAverage(vdResizeImage_ms);
        deviation = calcDeviation(vdResizeImage_ms, average);
        std::cout << "Image Resize: " << average << "$\\pm$" << deviation << std::endl;
        f << "Image Resize: " << average << "$\\pm$" << deviation << std::endl;
    }

    average = calcAverage(vdORBExtract_ms);
    deviation = calcDeviation(vdORBExtract_ms, average);
    std::cout << "ORB Extraction: " << average << "$\\pm$" << deviation << std::endl;
    f << "ORB Extraction: " << average << "$\\pm$" << deviation << std::endl;

    if(!vdStereoMatch_ms.empty())
    {
        average = calcAverage(vdStereoMatch_ms);
        deviation = calcDeviation(vdStereoMatch_ms, average);
        std::cout << "Stereo Matching: " << average << "$\\pm$" << deviation << std::endl;
        f << "Stereo Matching: " << average << "$\\pm$" << deviation << std::endl;
    }

    if(!vdIMUInteg_ms.empty())
    {
        average = calcAverage(vdIMUInteg_ms);
        deviation = calcDeviation(vdIMUInteg_ms, average);
        std::cout << "IMU Preintegration: " << average << "$\\pm$" << deviation << std::endl;
        f << "IMU Preintegration: " << average << "$\\pm$" << deviation << std::endl;
    }

    average = calcAverage(vdPosePred_ms);
    deviation = calcDeviation(vdPosePred_ms, average);
    std::cout << "Pose Prediction: " << average << "$\\pm$" << deviation << std::endl;
    f << "Pose Prediction: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(vdLMTrack_ms);
    deviation = calcDeviation(vdLMTrack_ms, average);
    std::cout << "LM Track: " << average << "$\\pm$" << deviation << std::endl;
    f << "LM Track: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(vdNewKF_ms);
    deviation = calcDeviation(vdNewKF_ms, average);
    std::cout << "New KF decision: " << average << "$\\pm$" << deviation << std::endl;
    f << "New KF decision: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(vdTrackTotal_ms);
    deviation = calcDeviation(vdTrackTotal_ms, average);
    std::cout << "Total Tracking: " << average << "$\\pm$" << deviation << std::endl;
    f << "Total Tracking: " << average << "$\\pm$" << deviation << std::endl;

    // Local Mapping time stats
    std::cout << std::endl << std::endl << std::endl;
    std::cout << "Local Mapping" << std::endl << std::endl;
    f << std::endl << "Local Mapping" << std::endl << std::endl;

    average = calcAverage(mpLocalMapper->vdKFInsert_ms);
    deviation = calcDeviation(mpLocalMapper->vdKFInsert_ms, average);
    std::cout << "KF Insertion: " << average << "$\\pm$" << deviation << std::endl;
    f << "KF Insertion: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vdMPCulling_ms);
    deviation = calcDeviation(mpLocalMapper->vdMPCulling_ms, average);
    std::cout << "MP Culling: " << average << "$\\pm$" << deviation << std::endl;
    f << "MP Culling: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vdMPCreation_ms);
    deviation = calcDeviation(mpLocalMapper->vdMPCreation_ms, average);
    std::cout << "MP Creation: " << average << "$\\pm$" << deviation << std::endl;
    f << "MP Creation: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vdLBA_ms);
    deviation = calcDeviation(mpLocalMapper->vdLBA_ms, average);
    std::cout << "LBA: " << average << "$\\pm$" << deviation << std::endl;
    f << "LBA: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vdKFCulling_ms);
    deviation = calcDeviation(mpLocalMapper->vdKFCulling_ms, average);
    std::cout << "KF Culling: " << average << "$\\pm$" << deviation << std::endl;
    f << "KF Culling: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vdLMTotal_ms);
    deviation = calcDeviation(mpLocalMapper->vdLMTotal_ms, average);
    std::cout << "Total Local Mapping: " << average << "$\\pm$" << deviation << std::endl;
    f << "Total Local Mapping: " << average << "$\\pm$" << deviation << std::endl;

    // Local Mapping LBA complexity
    std::cout << "---------------------------" << std::endl;
    std::cout << std::endl << "LBA complexity (mean$\\pm$std)" << std::endl;
    f << "---------------------------" << std::endl;
    f << std::endl << "LBA complexity (mean$\\pm$std)" << std::endl;

    average = calcAverage(mpLocalMapper->vnLBA_edges);
    deviation = calcDeviation(mpLocalMapper->vnLBA_edges, average);
    std::cout << "LBA Edges: " << average << "$\\pm$" << deviation << std::endl;
    f << "LBA Edges: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vnLBA_KFopt);
    deviation = calcDeviation(mpLocalMapper->vnLBA_KFopt, average);
    std::cout << "LBA KF optimized: " << average << "$\\pm$" << deviation << std::endl;
    f << "LBA KF optimized: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vnLBA_KFfixed);
    deviation = calcDeviation(mpLocalMapper->vnLBA_KFfixed, average);
    std::cout << "LBA KF fixed: " << average << "$\\pm$" << deviation << std::endl;
    f << "LBA KF fixed: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vnLBA_MPs);
    deviation = calcDeviation(mpLocalMapper->vnLBA_MPs, average);
    std::cout << "LBA MP: " << average << "$\\pm$" << deviation << std::endl << std::endl;
    f << "LBA MP: " << average << "$\\pm$" << deviation << std::endl << std::endl;

    std::cout << "LBA executions: " << mpLocalMapper->nLBA_exec << std::endl;
    std::cout << "LBA aborts: " << mpLocalMapper->nLBA_abort << std::endl;
    f << "LBA executions: " << mpLocalMapper->nLBA_exec << std::endl;
    f << "LBA aborts: " << mpLocalMapper->nLBA_abort << std::endl;

    // Map complexity
    std::cout << "---------------------------" << std::endl;
    std::cout << std::endl << "Map complexity" << std::endl;
    std::cout << "KFs in map: " << mpAtlas->GetAllKeyFrames().size() << std::endl;
    std::cout << "MPs in map: " << mpAtlas->GetAllMapPoints().size() << std::endl;
    f << "---------------------------" << std::endl;
    f << std::endl << "Map complexity" << std::endl;
    vector<Map*> vpMaps = mpAtlas->GetAllMaps();
    Map* pBestMap = vpMaps[0];
    for(int i=1; i<vpMaps.size(); ++i)
    {
        if(pBestMap->GetAllKeyFrames().size() < vpMaps[i]->GetAllKeyFrames().size())
        {
            pBestMap = vpMaps[i];
        }
    }

    f << "KFs in map: " << pBestMap->GetAllKeyFrames().size() << std::endl;
    f << "MPs in map: " << pBestMap->GetAllMapPoints().size() << std::endl;

    f << "---------------------------" << std::endl;
    f << std::endl << "Place Recognition (mean$\\pm$std)" << std::endl;
    std::cout << "---------------------------" << std::endl;
    std::cout << std::endl << "Place Recognition (mean$\\pm$std)" << std::endl;
    average = calcAverage(mpLoopClosing->vdDataQuery_ms);
    deviation = calcDeviation(mpLoopClosing->vdDataQuery_ms, average);
    f << "Database Query: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Database Query: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdEstSim3_ms);
    deviation = calcDeviation(mpLoopClosing->vdEstSim3_ms, average);
    f << "SE3 estimation: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "SE3 estimation: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdPRTotal_ms);
    deviation = calcDeviation(mpLoopClosing->vdPRTotal_ms, average);
    f << "Total Place Recognition: " << average << "$\\pm$" << deviation << std::endl << std::endl;
    std::cout << "Total Place Recognition: " << average << "$\\pm$" << deviation << std::endl << std::endl;

    f << std::endl << "Loop Closing (mean$\\pm$std)" << std::endl;
    std::cout << std::endl << "Loop Closing (mean$\\pm$std)" << std::endl;
    average = calcAverage(mpLoopClosing->vdLoopFusion_ms);
    deviation = calcDeviation(mpLoopClosing->vdLoopFusion_ms, average);
    f << "Loop Fusion: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Loop Fusion: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdLoopOptEss_ms);
    deviation = calcDeviation(mpLoopClosing->vdLoopOptEss_ms, average);
    f << "Essential Graph: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Essential Graph: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdLoopTotal_ms);
    deviation = calcDeviation(mpLoopClosing->vdLoopTotal_ms, average);
    f << "Total Loop Closing: " << average << "$\\pm$" << deviation << std::endl << std::endl;
    std::cout << "Total Loop Closing: " << average << "$\\pm$" << deviation << std::endl << std::endl;

    f << "Numb exec: " << mpLoopClosing->nLoop << std::endl;
    std::cout << "Num exec: " << mpLoopClosing->nLoop << std::endl;
    average = calcAverage(mpLoopClosing->vnLoopKFs);
    deviation = calcDeviation(mpLoopClosing->vnLoopKFs, average);
    f << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;

    f << std::endl << "Map Merging (mean$\\pm$std)" << std::endl;
    std::cout << std::endl << "Map Merging (mean$\\pm$std)" << std::endl;
    average = calcAverage(mpLoopClosing->vdMergeMaps_ms);
    deviation = calcDeviation(mpLoopClosing->vdMergeMaps_ms, average);
    f << "Merge Maps: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Merge Maps: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdWeldingBA_ms);
    deviation = calcDeviation(mpLoopClosing->vdWeldingBA_ms, average);
    f << "Welding BA: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Welding BA: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdMergeOptEss_ms);
    deviation = calcDeviation(mpLoopClosing->vdMergeOptEss_ms, average);
    f << "Optimization Ess.: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Optimization Ess.: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdMergeTotal_ms);
    deviation = calcDeviation(mpLoopClosing->vdMergeTotal_ms, average);
    f << "Total Map Merging: " << average << "$\\pm$" << deviation << std::endl << std::endl;
    std::cout << "Total Map Merging: " << average << "$\\pm$" << deviation << std::endl << std::endl;

    f << "Numb exec: " << mpLoopClosing->nMerges << std::endl;
    std::cout << "Num exec: " << mpLoopClosing->nMerges << std::endl;
    average = calcAverage(mpLoopClosing->vnMergeKFs);
    deviation = calcDeviation(mpLoopClosing->vnMergeKFs, average);
    f << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vnMergeMPs);
    deviation = calcDeviation(mpLoopClosing->vnMergeMPs, average);
    f << "Number of MPs: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Number of MPs: " << average << "$\\pm$" << deviation << std::endl;

    f << std::endl << "Full GBA (mean$\\pm$std)" << std::endl;
    std::cout << std::endl << "Full GBA (mean$\\pm$std)" << std::endl;
    average = calcAverage(mpLoopClosing->vdGBA_ms);
    deviation = calcDeviation(mpLoopClosing->vdGBA_ms, average);
    f << "GBA: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "GBA: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdUpdateMap_ms);
    deviation = calcDeviation(mpLoopClosing->vdUpdateMap_ms, average);
    f << "Map Update: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Map Update: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdFGBATotal_ms);
    deviation = calcDeviation(mpLoopClosing->vdFGBATotal_ms, average);
    f << "Total Full GBA: " << average << "$\\pm$" << deviation << std::endl << std::endl;
    std::cout << "Total Full GBA: " << average << "$\\pm$" << deviation << std::endl << std::endl;

    f << "Numb exec: " << mpLoopClosing->nFGBA_exec << std::endl;
    std::cout << "Num exec: " << mpLoopClosing->nFGBA_exec << std::endl;
    f << "Numb abort: " << mpLoopClosing->nFGBA_abort << std::endl;
    std::cout << "Num abort: " << mpLoopClosing->nFGBA_abort << std::endl;
    average = calcAverage(mpLoopClosing->vnGBAKFs);
    deviation = calcDeviation(mpLoopClosing->vnGBAKFs, average);
    f << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vnGBAMPs);
    deviation = calcDeviation(mpLoopClosing->vnGBAMPs, average);
    f << "Number of MPs: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Number of MPs: " << average << "$\\pm$" << deviation << std::endl;

    f.close();

}

#endif

// 析构函数
Tracking::~Tracking()
{
    //f_track_stats.close();

}



// 配置参数加载，解析 settings
void Tracking::newParameterLoader(Settings *settings) {
    mpCamera = settings->camera1();          // 获取相机 1 内参
    mpCamera = mpAtlas->AddCamera(mpCamera); // 在 Atlas 中注册相机 1

    if(settings->needToUndistort()){ // 是否去畸变（单目）
        mDistCoef = settings->camera1DistortionCoef();
    }
    else{
        mDistCoef = cv::Mat::zeros(4,1,CV_32F);
    }

    //TODO: missing image scaling and rectification
    mImageScale = 1.0f;

    mK = cv::Mat::eye(3,3,CV_32F); // opencv 格式相机内参
    mK.at<float>(0,0) = mpCamera->getParameter(0);
    mK.at<float>(1,1) = mpCamera->getParameter(1);
    mK.at<float>(0,2) = mpCamera->getParameter(2);
    mK.at<float>(1,2) = mpCamera->getParameter(3);

    mK_.setIdentity(); // eigen 格式相机内参
    mK_(0,0) = mpCamera->getParameter(0);
    mK_(1,1) = mpCamera->getParameter(1);
    mK_(0,2) = mpCamera->getParameter(2);
    mK_(1,2) = mpCamera->getParameter(3);

    if((mSensor==System::STEREO || mSensor==System::IMU_STEREO || mSensor==System::IMU_RGBD) &&
        settings->cameraType() == Settings::KannalaBrandt){ // 双目/ RGB-D 且为鱼眼相机
        mpCamera2 = settings->camera2();           // 获取相机 2 内参
        mpCamera2 = mpAtlas->AddCamera(mpCamera2); // 在 Atlas 中注册相机 2

        mTlr = settings->Tlr(); // 相机外参

        mpFrameDrawer->both = true; // 同时绘制左右两图像
    }

    if(mSensor==System::STEREO || mSensor==System::RGBD || mSensor==System::IMU_STEREO || mSensor==System::IMU_RGBD ){
        mbf = settings->bf();                           // 获取基线*焦距
        mThDepth = settings->b() * settings->thDepth(); // 深度阈值
    }

    if(mSensor==System::RGBD || mSensor==System::IMU_RGBD){
        mDepthMapFactor = settings->depthMapFactor(); // 深度图因子
        if(fabs(mDepthMapFactor)<1e-5)
            mDepthMapFactor=1;
        else
            mDepthMapFactor = 1.0f/mDepthMapFactor;
    }

    mMinFrames = 0;               // 最小关键帧间隔，取 0
    mMaxFrames = settings->fps(); // 最大关键帧间隔，取帧率值
    mbRGB = settings->rgb();      // RGB 颜色顺序

    //ORB parameters
    int nFeatures = settings->nFeatures();        // 特征点数量
    int nLevels = settings->nLevels();            // 金字塔层数
    int fIniThFAST = settings->initThFAST();      // 初始 FAST
    int fMinThFAST = settings->minThFAST();       // 最小 FAST
    float fScaleFactor = settings->scaleFactor(); // 缩放系数

    mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST); // 左图像特征提取器

    if(mSensor==System::STEREO || mSensor==System::IMU_STEREO) // 右图像特征提取器（双目）
        mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR) // 初始化专用特征提取器（单目）
        mpIniORBextractor = new ORBextractor(5*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    //IMU parameters
    Sophus::SE3f Tbc = settings->Tbc();             // 相机相对于 IMU 的位姿
    mInsertKFsLost = settings->insertKFsWhenLost(); // 丢失时是否插入关键帧
    mImuFreq = settings->imuFrequency();            // IMU 采样率
    mImuPer = 0.001;                                // IMU 周期      1.0 / (double) mImuFreq; TODO: ESTO ESTA BIEN?
    float Ng = settings->noiseGyro();               // 陀螺仪噪声
    float Na = settings->noiseAcc();                // 加速度计噪声
    float Ngw = settings->gyroWalk();               // 陀螺仪随机游走
    float Naw = settings->accWalk();                // 加速度计随机游走

    const float sf = sqrt(mImuFreq);                            // 采样率平方根
    mpImuCalib = new IMU::Calib(Tbc,Ng*sf,Na*sf,Ngw/sf,Naw/sf); // IMU 标定参数对象

    mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib); // 创建上一关键帧到当前帧的 IMU 预积分对象
}



// 解析相机参数
bool Tracking::ParseCamParamFile(cv::FileStorage &fSettings)
{
    mDistCoef = cv::Mat::zeros(4,1,CV_32F);
    cout << endl << "Camera Parameters: " << endl;
    bool b_miss_params = false;

    string sCameraName = fSettings["Camera.type"];
    if(sCameraName == "PinHole")
    {
        float fx, fy, cx, cy;
        mImageScale = 1.f;

        // Camera calibration parameters
        cv::FileNode node = fSettings["Camera.fx"];
        if(!node.empty() && node.isReal())
        {
            fx = node.real();
        }
        else
        {
            std::cerr << "*Camera.fx parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.fy"];
        if(!node.empty() && node.isReal())
        {
            fy = node.real();
        }
        else
        {
            std::cerr << "*Camera.fy parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.cx"];
        if(!node.empty() && node.isReal())
        {
            cx = node.real();
        }
        else
        {
            std::cerr << "*Camera.cx parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.cy"];
        if(!node.empty() && node.isReal())
        {
            cy = node.real();
        }
        else
        {
            std::cerr << "*Camera.cy parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        // Distortion parameters
        node = fSettings["Camera.k1"];
        if(!node.empty() && node.isReal())
        {
            mDistCoef.at<float>(0) = node.real();
        }
        else
        {
            std::cerr << "*Camera.k1 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.k2"];
        if(!node.empty() && node.isReal())
        {
            mDistCoef.at<float>(1) = node.real();
        }
        else
        {
            std::cerr << "*Camera.k2 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.p1"];
        if(!node.empty() && node.isReal())
        {
            mDistCoef.at<float>(2) = node.real();
        }
        else
        {
            std::cerr << "*Camera.p1 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.p2"];
        if(!node.empty() && node.isReal())
        {
            mDistCoef.at<float>(3) = node.real();
        }
        else
        {
            std::cerr << "*Camera.p2 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.k3"];
        if(!node.empty() && node.isReal())
        {
            mDistCoef.resize(5);
            mDistCoef.at<float>(4) = node.real();
        }

        node = fSettings["Camera.imageScale"];
        if(!node.empty() && node.isReal())
        {
            mImageScale = node.real();
        }

        if(b_miss_params)
        {
            return false;
        }

        if(mImageScale != 1.f)
        {
            // K matrix parameters must be scaled.
            fx = fx * mImageScale;
            fy = fy * mImageScale;
            cx = cx * mImageScale;
            cy = cy * mImageScale;
        }

        vector<float> vCamCalib{fx,fy,cx,cy};

        mpCamera = new Pinhole(vCamCalib);

        mpCamera = mpAtlas->AddCamera(mpCamera);

        std::cout << "- Camera: Pinhole" << std::endl;
        std::cout << "- Image scale: " << mImageScale << std::endl;
        std::cout << "- fx: " << fx << std::endl;
        std::cout << "- fy: " << fy << std::endl;
        std::cout << "- cx: " << cx << std::endl;
        std::cout << "- cy: " << cy << std::endl;
        std::cout << "- k1: " << mDistCoef.at<float>(0) << std::endl;
        std::cout << "- k2: " << mDistCoef.at<float>(1) << std::endl;


        std::cout << "- p1: " << mDistCoef.at<float>(2) << std::endl;
        std::cout << "- p2: " << mDistCoef.at<float>(3) << std::endl;

        if(mDistCoef.rows==5)
            std::cout << "- k3: " << mDistCoef.at<float>(4) << std::endl;

        mK = cv::Mat::eye(3,3,CV_32F);
        mK.at<float>(0,0) = fx;
        mK.at<float>(1,1) = fy;
        mK.at<float>(0,2) = cx;
        mK.at<float>(1,2) = cy;

        mK_.setIdentity();
        mK_(0,0) = fx;
        mK_(1,1) = fy;
        mK_(0,2) = cx;
        mK_(1,2) = cy;
    }
    else if(sCameraName == "KannalaBrandt8")
    {
        float fx, fy, cx, cy;
        float k1, k2, k3, k4;
        mImageScale = 1.f;

        // Camera calibration parameters
        cv::FileNode node = fSettings["Camera.fx"];
        if(!node.empty() && node.isReal())
        {
            fx = node.real();
        }
        else
        {
            std::cerr << "*Camera.fx parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }
        node = fSettings["Camera.fy"];
        if(!node.empty() && node.isReal())
        {
            fy = node.real();
        }
        else
        {
            std::cerr << "*Camera.fy parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.cx"];
        if(!node.empty() && node.isReal())
        {
            cx = node.real();
        }
        else
        {
            std::cerr << "*Camera.cx parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.cy"];
        if(!node.empty() && node.isReal())
        {
            cy = node.real();
        }
        else
        {
            std::cerr << "*Camera.cy parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        // Distortion parameters
        node = fSettings["Camera.k1"];
        if(!node.empty() && node.isReal())
        {
            k1 = node.real();
        }
        else
        {
            std::cerr << "*Camera.k1 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }
        node = fSettings["Camera.k2"];
        if(!node.empty() && node.isReal())
        {
            k2 = node.real();
        }
        else
        {
            std::cerr << "*Camera.k2 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.k3"];
        if(!node.empty() && node.isReal())
        {
            k3 = node.real();
        }
        else
        {
            std::cerr << "*Camera.k3 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.k4"];
        if(!node.empty() && node.isReal())
        {
            k4 = node.real();
        }
        else
        {
            std::cerr << "*Camera.k4 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.imageScale"];
        if(!node.empty() && node.isReal())
        {
            mImageScale = node.real();
        }

        if(!b_miss_params)
        {
            if(mImageScale != 1.f)
            {
                // K matrix parameters must be scaled.
                fx = fx * mImageScale;
                fy = fy * mImageScale;
                cx = cx * mImageScale;
                cy = cy * mImageScale;
            }

            vector<float> vCamCalib{fx,fy,cx,cy,k1,k2,k3,k4};
            mpCamera = new KannalaBrandt8(vCamCalib);
            mpCamera = mpAtlas->AddCamera(mpCamera);
            std::cout << "- Camera: Fisheye" << std::endl;
            std::cout << "- Image scale: " << mImageScale << std::endl;
            std::cout << "- fx: " << fx << std::endl;
            std::cout << "- fy: " << fy << std::endl;
            std::cout << "- cx: " << cx << std::endl;
            std::cout << "- cy: " << cy << std::endl;
            std::cout << "- k1: " << k1 << std::endl;
            std::cout << "- k2: " << k2 << std::endl;
            std::cout << "- k3: " << k3 << std::endl;
            std::cout << "- k4: " << k4 << std::endl;

            mK = cv::Mat::eye(3,3,CV_32F);
            mK.at<float>(0,0) = fx;
            mK.at<float>(1,1) = fy;
            mK.at<float>(0,2) = cx;
            mK.at<float>(1,2) = cy;

            mK_.setIdentity();
            mK_(0,0) = fx;
            mK_(1,1) = fy;
            mK_(0,2) = cx;
            mK_(1,2) = cy;
        }

        if(mSensor==System::STEREO || mSensor==System::IMU_STEREO || mSensor==System::IMU_RGBD){
            // Right camera
            // Camera calibration parameters
            cv::FileNode node = fSettings["Camera2.fx"];
            if(!node.empty() && node.isReal())
            {
                fx = node.real();
            }
            else
            {
                std::cerr << "*Camera2.fx parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }
            node = fSettings["Camera2.fy"];
            if(!node.empty() && node.isReal())
            {
                fy = node.real();
            }
            else
            {
                std::cerr << "*Camera2.fy parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera2.cx"];
            if(!node.empty() && node.isReal())
            {
                cx = node.real();
            }
            else
            {
                std::cerr << "*Camera2.cx parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera2.cy"];
            if(!node.empty() && node.isReal())
            {
                cy = node.real();
            }
            else
            {
                std::cerr << "*Camera2.cy parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            // Distortion parameters
            node = fSettings["Camera2.k1"];
            if(!node.empty() && node.isReal())
            {
                k1 = node.real();
            }
            else
            {
                std::cerr << "*Camera2.k1 parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }
            node = fSettings["Camera2.k2"];
            if(!node.empty() && node.isReal())
            {
                k2 = node.real();
            }
            else
            {
                std::cerr << "*Camera2.k2 parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera2.k3"];
            if(!node.empty() && node.isReal())
            {
                k3 = node.real();
            }
            else
            {
                std::cerr << "*Camera2.k3 parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera2.k4"];
            if(!node.empty() && node.isReal())
            {
                k4 = node.real();
            }
            else
            {
                std::cerr << "*Camera2.k4 parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }


            int leftLappingBegin = -1;
            int leftLappingEnd = -1;

            int rightLappingBegin = -1;
            int rightLappingEnd = -1;

            node = fSettings["Camera.lappingBegin"];
            if(!node.empty() && node.isInt())
            {
                leftLappingBegin = node.operator int();
            }
            else
            {
                std::cout << "WARNING: Camera.lappingBegin not correctly defined" << std::endl;
            }
            node = fSettings["Camera.lappingEnd"];
            if(!node.empty() && node.isInt())
            {
                leftLappingEnd = node.operator int();
            }
            else
            {
                std::cout << "WARNING: Camera.lappingEnd not correctly defined" << std::endl;
            }
            node = fSettings["Camera2.lappingBegin"];
            if(!node.empty() && node.isInt())
            {
                rightLappingBegin = node.operator int();
            }
            else
            {
                std::cout << "WARNING: Camera2.lappingBegin not correctly defined" << std::endl;
            }
            node = fSettings["Camera2.lappingEnd"];
            if(!node.empty() && node.isInt())
            {
                rightLappingEnd = node.operator int();
            }
            else
            {
                std::cout << "WARNING: Camera2.lappingEnd not correctly defined" << std::endl;
            }

            node = fSettings["Tlr"];
            cv::Mat cvTlr;
            if(!node.empty())
            {
                cvTlr = node.mat();
                if(cvTlr.rows != 3 || cvTlr.cols != 4)
                {
                    std::cerr << "*Tlr matrix have to be a 3x4 transformation matrix*" << std::endl;
                    b_miss_params = true;
                }
            }
            else
            {
                std::cerr << "*Tlr matrix doesn't exist*" << std::endl;
                b_miss_params = true;
            }

            if(!b_miss_params)
            {
                if(mImageScale != 1.f)
                {
                    // K matrix parameters must be scaled.
                    fx = fx * mImageScale;
                    fy = fy * mImageScale;
                    cx = cx * mImageScale;
                    cy = cy * mImageScale;

                    leftLappingBegin = leftLappingBegin * mImageScale;
                    leftLappingEnd = leftLappingEnd * mImageScale;
                    rightLappingBegin = rightLappingBegin * mImageScale;
                    rightLappingEnd = rightLappingEnd * mImageScale;
                }

                static_cast<KannalaBrandt8*>(mpCamera)->mvLappingArea[0] = leftLappingBegin;
                static_cast<KannalaBrandt8*>(mpCamera)->mvLappingArea[1] = leftLappingEnd;

                mpFrameDrawer->both = true;

                vector<float> vCamCalib2{fx,fy,cx,cy,k1,k2,k3,k4};
                mpCamera2 = new KannalaBrandt8(vCamCalib2);
                mpCamera2 = mpAtlas->AddCamera(mpCamera2);

                mTlr = Converter::toSophus(cvTlr);

                static_cast<KannalaBrandt8*>(mpCamera2)->mvLappingArea[0] = rightLappingBegin;
                static_cast<KannalaBrandt8*>(mpCamera2)->mvLappingArea[1] = rightLappingEnd;

                std::cout << "- Camera1 Lapping: " << leftLappingBegin << ", " << leftLappingEnd << std::endl;

                std::cout << std::endl << "Camera2 Parameters:" << std::endl;
                std::cout << "- Camera: Fisheye" << std::endl;
                std::cout << "- Image scale: " << mImageScale << std::endl;
                std::cout << "- fx: " << fx << std::endl;
                std::cout << "- fy: " << fy << std::endl;
                std::cout << "- cx: " << cx << std::endl;
                std::cout << "- cy: " << cy << std::endl;
                std::cout << "- k1: " << k1 << std::endl;
                std::cout << "- k2: " << k2 << std::endl;
                std::cout << "- k3: " << k3 << std::endl;
                std::cout << "- k4: " << k4 << std::endl;

                std::cout << "- mTlr: \n" << cvTlr << std::endl;

                std::cout << "- Camera2 Lapping: " << rightLappingBegin << ", " << rightLappingEnd << std::endl;
            }
        }

        if(b_miss_params)
        {
            return false;
        }

    }
    else
    {
        std::cerr << "*Not Supported Camera Sensor*" << std::endl;
        std::cerr << "Check an example configuration file with the desired sensor" << std::endl;
    }

    if(mSensor==System::STEREO || mSensor==System::RGBD || mSensor==System::IMU_STEREO || mSensor==System::IMU_RGBD )
    {
        cv::FileNode node = fSettings["Camera.bf"];
        if(!node.empty() && node.isReal())
        {
            mbf = node.real();
            if(mImageScale != 1.f)
            {
                mbf *= mImageScale;
            }
        }
        else
        {
            std::cerr << "*Camera.bf parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

    }

    float fps = fSettings["Camera.fps"];
    if(fps==0)
        fps=30;

    // Max/Min Frames to insert keyframes and to check relocalisation
    mMinFrames = 0;
    mMaxFrames = fps;

    cout << "- fps: " << fps << endl;


    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    if(mbRGB)
        cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
        cout << "- color order: BGR (ignored if grayscale)" << endl;

    if(mSensor==System::STEREO || mSensor==System::RGBD || mSensor==System::IMU_STEREO || mSensor==System::IMU_RGBD)
    {
        float fx = mpCamera->getParameter(0);
        cv::FileNode node = fSettings["ThDepth"];
        if(!node.empty()  && node.isReal())
        {
            mThDepth = node.real();
            mThDepth = mbf*mThDepth/fx;
            cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
        }
        else
        {
            std::cerr << "*ThDepth parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }


    }

    if(mSensor==System::RGBD || mSensor==System::IMU_RGBD)
    {
        cv::FileNode node = fSettings["DepthMapFactor"];
        if(!node.empty() && node.isReal())
        {
            mDepthMapFactor = node.real();
            if(fabs(mDepthMapFactor)<1e-5)
                mDepthMapFactor=1;
            else
                mDepthMapFactor = 1.0f/mDepthMapFactor;
        }
        else
        {
            std::cerr << "*DepthMapFactor parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

    }

    if(b_miss_params)
    {
        return false;
    }

    return true;
}

// 解析 ORB 特征提取参数
bool Tracking::ParseORBParamFile(cv::FileStorage &fSettings)
{
    bool b_miss_params = false;
    int nFeatures, nLevels, fIniThFAST, fMinThFAST;
    float fScaleFactor;

    cv::FileNode node = fSettings["ORBextractor.nFeatures"];
    if(!node.empty() && node.isInt())
    {
        nFeatures = node.operator int();
    }
    else
    {
        std::cerr << "*ORBextractor.nFeatures parameter doesn't exist or is not an integer*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["ORBextractor.scaleFactor"];
    if(!node.empty() && node.isReal())
    {
        fScaleFactor = node.real();
    }
    else
    {
        std::cerr << "*ORBextractor.scaleFactor parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["ORBextractor.nLevels"];
    if(!node.empty() && node.isInt())
    {
        nLevels = node.operator int();
    }
    else
    {
        std::cerr << "*ORBextractor.nLevels parameter doesn't exist or is not an integer*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["ORBextractor.iniThFAST"];
    if(!node.empty() && node.isInt())
    {
        fIniThFAST = node.operator int();
    }
    else
    {
        std::cerr << "*ORBextractor.iniThFAST parameter doesn't exist or is not an integer*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["ORBextractor.minThFAST"];
    if(!node.empty() && node.isInt())
    {
        fMinThFAST = node.operator int();
    }
    else
    {
        std::cerr << "*ORBextractor.minThFAST parameter doesn't exist or is not an integer*" << std::endl;
        b_miss_params = true;
    }

    if(b_miss_params)
    {
        return false;
    }

    mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(mSensor==System::STEREO || mSensor==System::IMU_STEREO)
        mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR)
        mpIniORBextractor = new ORBextractor(5*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    cout << endl << "ORB Extractor Parameters: " << endl;
    cout << "- Number of Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;
    cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
    cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

    return true;
}

// 解析 IMU 参数
bool Tracking::ParseIMUParamFile(cv::FileStorage &fSettings)
{
    bool b_miss_params = false;

    cv::Mat cvTbc;
    cv::FileNode node = fSettings["Tbc"];
    if(!node.empty())
    {
        cvTbc = node.mat();
        if(cvTbc.rows != 4 || cvTbc.cols != 4)
        {
            std::cerr << "*Tbc matrix have to be a 4x4 transformation matrix*" << std::endl;
            b_miss_params = true;
        }
    }
    else
    {
        std::cerr << "*Tbc matrix doesn't exist*" << std::endl;
        b_miss_params = true;
    }
    cout << endl;
    cout << "Left camera to Imu Transform (Tbc): " << endl << cvTbc << endl;
    Eigen::Matrix<float,4,4,Eigen::RowMajor> eigTbc(cvTbc.ptr<float>(0));
    Sophus::SE3f Tbc(eigTbc);

    node = fSettings["InsertKFsWhenLost"];
    mInsertKFsLost = true;
    if(!node.empty() && node.isInt())
    {
        mInsertKFsLost = (bool) node.operator int();
    }

    if(!mInsertKFsLost)
        cout << "Do not insert keyframes when lost visual tracking " << endl;



    float Ng, Na, Ngw, Naw;

    node = fSettings["IMU.Frequency"];
    if(!node.empty() && node.isInt())
    {
        mImuFreq = node.operator int();
        mImuPer = 0.001; //1.0 / (double) mImuFreq;
    }
    else
    {
        std::cerr << "*IMU.Frequency parameter doesn't exist or is not an integer*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["IMU.NoiseGyro"];
    if(!node.empty() && node.isReal())
    {
        Ng = node.real();
    }
    else
    {
        std::cerr << "*IMU.NoiseGyro parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["IMU.NoiseAcc"];
    if(!node.empty() && node.isReal())
    {
        Na = node.real();
    }
    else
    {
        std::cerr << "*IMU.NoiseAcc parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["IMU.GyroWalk"];
    if(!node.empty() && node.isReal())
    {
        Ngw = node.real();
    }
    else
    {
        std::cerr << "*IMU.GyroWalk parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["IMU.AccWalk"];
    if(!node.empty() && node.isReal())
    {
        Naw = node.real();
    }
    else
    {
        std::cerr << "*IMU.AccWalk parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["IMU.fastInit"];
    mFastInit = false;
    if(!node.empty())
    {
        mFastInit = static_cast<int>(fSettings["IMU.fastInit"]) != 0;
    }

    if(mFastInit)
        cout << "Fast IMU initialization. Acceleration is not checked \n";

    if(b_miss_params)
    {
        return false;
    }

    const float sf = sqrt(mImuFreq);
    cout << endl;
    cout << "IMU frequency: " << mImuFreq << " Hz" << endl;
    cout << "IMU gyro noise: " << Ng << " rad/s/sqrt(Hz)" << endl;
    cout << "IMU gyro walk: " << Ngw << " rad/s^2/sqrt(Hz)" << endl;
    cout << "IMU accelerometer noise: " << Na << " m/s^2/sqrt(Hz)" << endl;
    cout << "IMU accelerometer walk: " << Naw << " m/s^3/sqrt(Hz)" << endl;

    mpImuCalib = new IMU::Calib(Tbc,Ng*sf,Na*sf,Ngw/sf,Naw/sf);

    mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);


    return true;
}



// 指向局部建图线程的指针
void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}

// 指向回环检测线程的指针
void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
{
    mpLoopClosing=pLoopClosing;
}

// 指向可视化线程的指针
void Tracking::SetViewer(Viewer *pViewer)
{
    mpViewer=pViewer;
}

// 单步调试模式设置
void Tracking::SetStepByStep(bool bSet)
{
    bStepByStep = bSet;
}

// 单步调试模式状态
bool Tracking::GetStepByStep()
{
    return bStepByStep;
}



// 双目图像抓取和处理，并调用 Track() 进行跟踪
Sophus::SE3f Tracking::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp, string filename)
{
    //cout << "GrabImageStereo" << endl;

    mImGray = imRectLeft;
    cv::Mat imGrayRight = imRectRight;
    mImRight = imRectRight;

    if(mImGray.channels()==3) // RGB 或 BGR 转灰度
    {
        //cout << "Image with 3 channels" << endl;
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,cv::COLOR_RGB2GRAY);
            cvtColor(imGrayRight,imGrayRight,cv::COLOR_RGB2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,cv::COLOR_BGR2GRAY);
            cvtColor(imGrayRight,imGrayRight,cv::COLOR_BGR2GRAY);
        }
    }
    else if(mImGray.channels()==4) // RGBA 或 BGRA 转灰度
    {
        //cout << "Image with 4 channels" << endl;
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,cv::COLOR_RGBA2GRAY);
            cvtColor(imGrayRight,imGrayRight,cv::COLOR_RGBA2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,cv::COLOR_BGRA2GRAY);
            cvtColor(imGrayRight,imGrayRight,cv::COLOR_BGRA2GRAY);
        }
    }

    //cout << "Incoming frame creation" << endl;
    // 创建当前帧对象
    if (mSensor == System::STEREO && !mpCamera2) // 双目且无 mpCamera2（非鱼眼双目，详见 Tracking::newParameterLoader ）
        mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera);
    else if(mSensor == System::STEREO && mpCamera2) // 双目且有 mpCamera2
        mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera,mpCamera2,mTlr);
    else if(mSensor == System::IMU_STEREO && !mpCamera2) // 双目惯性且无 mpCamera2
        mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera,&mLastFrame,*mpImuCalib);
    else if(mSensor == System::IMU_STEREO && mpCamera2) // 双目惯性且有 mpCamera2
        mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera,mpCamera2,mTlr,&mLastFrame,*mpImuCalib);

    //cout << "Incoming frame ended" << endl;

    mCurrentFrame.mNameFile = filename;     // filename 可为空（System::Track* 里的调用传入默认为空）
    mCurrentFrame.mnDataset = mnNumDataset; // 数据集编号

#ifdef REGISTER_TIMES // 时间统计（REGISTER_TIMES 宏定义打开时，位于 include/Setting.h 中）
    vdORBExtract_ms.push_back(mCurrentFrame.mTimeORB_Ext);      // 特征提取耗时
    vdStereoMatch_ms.push_back(mCurrentFrame.mTimeStereoMatch); // 双目匹配耗时
#endif

    //cout << "Tracking start" << endl;
    Track(); // 跟踪线程主函数
    //cout << "Tracking end" << endl;

    return mCurrentFrame.GetPose(); // 返回当前帧的相机位姿
}

// RGB-D 图像抓取和处理，并调用 Track() 进行跟踪
Sophus::SE3f Tracking::GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp, string filename)
{
    mImGray = imRGB;
    cv::Mat imDepth = imD;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,cv::COLOR_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,cv::COLOR_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,cv::COLOR_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,cv::COLOR_BGRA2GRAY);
    }

    if((fabs(mDepthMapFactor-1.0f)>1e-5) || imDepth.type()!=CV_32F)
        imDepth.convertTo(imDepth,CV_32F,mDepthMapFactor);

    if (mSensor == System::RGBD)
        mCurrentFrame = Frame(mImGray,imDepth,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera);
    else if(mSensor == System::IMU_RGBD)
        mCurrentFrame = Frame(mImGray,imDepth,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera,&mLastFrame,*mpImuCalib);






    mCurrentFrame.mNameFile = filename;
    mCurrentFrame.mnDataset = mnNumDataset;

#ifdef REGISTER_TIMES
    vdORBExtract_ms.push_back(mCurrentFrame.mTimeORB_Ext);
#endif

    Track();

    return mCurrentFrame.GetPose();
}

// 单目图像抓取和处理，并调用 Track() 进行跟踪
Sophus::SE3f Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp, string filename)
{
    mImGray = im;
    if(mImGray.channels()==3) // RGB 或 BGR 转灰度
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,cv::COLOR_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,cv::COLOR_BGR2GRAY);
    }
    else if(mImGray.channels()==4) // RGBA 或 BGRA 转灰度
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,cv::COLOR_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,cv::COLOR_BGRA2GRAY);
    }
    
    // 创建当前帧对象
    if (mSensor == System::MONOCULAR) // 单目
    {
        if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET ||(lastID - initID) < mMaxFrames) // 未初始化
            mCurrentFrame = Frame(mImGray,timestamp,mpIniORBextractor,mpORBVocabulary,mpCamera,mDistCoef,mbf,mThDepth);
        else // 已初始化
            mCurrentFrame = Frame(mImGray,timestamp,mpORBextractorLeft,mpORBVocabulary,mpCamera,mDistCoef,mbf,mThDepth);
    }
    else if(mSensor == System::IMU_MONOCULAR) // 单目惯性
    {
        if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET) // 未初始化
        {
            mCurrentFrame = Frame(mImGray,timestamp,mpIniORBextractor,mpORBVocabulary,mpCamera,mDistCoef,mbf,mThDepth,&mLastFrame,*mpImuCalib);
        }
        else // 已初始化
            mCurrentFrame = Frame(mImGray,timestamp,mpORBextractorLeft,mpORBVocabulary,mpCamera,mDistCoef,mbf,mThDepth,&mLastFrame,*mpImuCalib);
    }

    if (mState==NO_IMAGES_YET) // 记录第一帧图像的时间戳
        t0=timestamp;

    mCurrentFrame.mNameFile = filename;     // filename 可为空（System::Track* 里的调用传入默认为空）
    mCurrentFrame.mnDataset = mnNumDataset; // 数据集编号

#ifdef REGISTER_TIMES // 时间统计（REGISTER_TIMES 宏定义打开时，位于 include/Setting.h 中）
    vdORBExtract_ms.push_back(mCurrentFrame.mTimeORB_Ext); // 特征提取耗时
#endif

    lastID = mCurrentFrame.mnId; // 更新最后帧 ID
    Track(); // 跟踪线程主函数

    return mCurrentFrame.GetPose(); // 返回当前帧的相机位姿
}



// IMU 数据抓取
void Tracking::GrabImuData(const IMU::Point &imuMeasurement)
{
    unique_lock<mutex> lock(mMutexImuQueue);
    mlQueueImuData.push_back(imuMeasurement);
}

// 帧间 IMU 数据预积分
void Tracking::PreintegrateIMU()
{

    if(!mCurrentFrame.mpPrevFrame) // 上一帧是否存在
    {
        Verbose::PrintMess("non prev frame ", Verbose::VERBOSITY_NORMAL);
        mCurrentFrame.setIntegrated();
        return;
    }

    mvImuFromLastFrame.clear(); // 清空
    mvImuFromLastFrame.reserve(mlQueueImuData.size()); // 预分配内存
    if(mlQueueImuData.size() == 0) // IMU 数据队列为空
    {
        Verbose::PrintMess("Not IMU data in mlQueueImuData!!", Verbose::VERBOSITY_NORMAL);
        mCurrentFrame.setIntegrated();
        return;
    }

    while(true)
    {
        bool bSleep = false; // 睡眠标志
        {
            unique_lock<mutex> lock(mMutexImuQueue); // IMU 队列保护锁
            if(!mlQueueImuData.empty()) // IMU 队列非空
            {
                IMU::Point* m = &mlQueueImuData.front(); // 取 IMU 队列首个元素
                cout.precision(17); // 设置输出精度
                if(m->t<mCurrentFrame.mpPrevFrame->mTimeStamp-mImuPer) // 首个元素的时间早于上一帧（IMU 时间 < 上一帧时间 - IMU 采样时间间隔）
                {
                    mlQueueImuData.pop_front(); // 丢弃当前 IMU 队列的首个元素
                }
                else if(m->t<mCurrentFrame.mTimeStamp-mImuPer) // 首个元素的时间在上一帧与当前帧之间（IMU 时间 < 当前帧时间 - IMU 采样时间间隔）
                {
                    mvImuFromLastFrame.push_back(*m); // 复制 IMU 队列的首个元素
                    mlQueueImuData.pop_front();       // 丢弃当前 IMU 队列的首个元素
                }
                else // 首个元素的时间接近或大于当前帧（IMU 时间 > 当前帧时间 - IMU 采样时间间隔）
                {
                    mvImuFromLastFrame.push_back(*m); // 复制 IMU 队列的首个元素
                    break; // 退出循环
                }
            }
            else // IMU 队列为空
            {
                break; // 退出循环
                bSleep = true; // 启动休眠
            }
        }
        if(bSleep)
            usleep(500); // 休眠 500 us（实际上已 break 退出循环，该 if 不会生效）
    }

    const int n = mvImuFromLastFrame.size()-1; // 已记录的帧间 IMU 数据量 -1
    if(n==0){ // 已记录的帧间 IMU 数据量仅有 1 或 0 条
        cout << "Empty IMU measurements vector!!!\n";
        return;
    }

    // 创建预积分对象
    IMU::Preintegrated* pImuPreintegratedFromLastFrame = new IMU::Preintegrated(mLastFrame.mImuBias,mCurrentFrame.mImuCalib);

    for(int i=0; i<n; i++) // 预积分
    {
        float tstep;
        Eigen::Vector3f acc, angVel;
        if((i==0) && (i<(n-1)))
        {
            float tab = mvImuFromLastFrame[i+1].t-mvImuFromLastFrame[i].t;
            float tini = mvImuFromLastFrame[i].t-mCurrentFrame.mpPrevFrame->mTimeStamp;
            acc = (mvImuFromLastFrame[i].a+mvImuFromLastFrame[i+1].a-
                    (mvImuFromLastFrame[i+1].a-mvImuFromLastFrame[i].a)*(tini/tab))*0.5f;
            angVel = (mvImuFromLastFrame[i].w+mvImuFromLastFrame[i+1].w-
                    (mvImuFromLastFrame[i+1].w-mvImuFromLastFrame[i].w)*(tini/tab))*0.5f;
            tstep = mvImuFromLastFrame[i+1].t-mCurrentFrame.mpPrevFrame->mTimeStamp;
        }
        else if(i<(n-1))
        {
            acc = (mvImuFromLastFrame[i].a+mvImuFromLastFrame[i+1].a)*0.5f;
            angVel = (mvImuFromLastFrame[i].w+mvImuFromLastFrame[i+1].w)*0.5f;
            tstep = mvImuFromLastFrame[i+1].t-mvImuFromLastFrame[i].t;
        }
        else if((i>0) && (i==(n-1)))
        {
            float tab = mvImuFromLastFrame[i+1].t-mvImuFromLastFrame[i].t;
            float tend = mvImuFromLastFrame[i+1].t-mCurrentFrame.mTimeStamp;
            acc = (mvImuFromLastFrame[i].a+mvImuFromLastFrame[i+1].a-
                    (mvImuFromLastFrame[i+1].a-mvImuFromLastFrame[i].a)*(tend/tab))*0.5f;
            angVel = (mvImuFromLastFrame[i].w+mvImuFromLastFrame[i+1].w-
                    (mvImuFromLastFrame[i+1].w-mvImuFromLastFrame[i].w)*(tend/tab))*0.5f;
            tstep = mCurrentFrame.mTimeStamp-mvImuFromLastFrame[i].t;
        }
        else if((i==0) && (i==(n-1)))
        {
            acc = mvImuFromLastFrame[i].a;
            angVel = mvImuFromLastFrame[i].w;
            tstep = mCurrentFrame.mTimeStamp-mCurrentFrame.mpPrevFrame->mTimeStamp;
        }

        if (!mpImuPreintegratedFromLastKF)
            cout << "mpImuPreintegratedFromLastKF does not exist" << endl;
        mpImuPreintegratedFromLastKF->IntegrateNewMeasurement(acc,angVel,tstep);
        pImuPreintegratedFromLastFrame->IntegrateNewMeasurement(acc,angVel,tstep);
    }

    mCurrentFrame.mpImuPreintegratedFrame = pImuPreintegratedFromLastFrame; // 从上一帧到当前帧的预积分
    mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;        // 从上一关键帧到当前帧的累积预积分
    mCurrentFrame.mpLastKeyFrame = mpLastKeyFrame;                          // 当前帧关联的上一关键帧

    mCurrentFrame.setIntegrated(); // IMU 预积分已完成

    //Verbose::PrintMess("Preintegration is finished!! ", Verbose::VERBOSITY_DEBUG);
}

// 用 IMU 预积分做位姿估计
bool Tracking::PredictStateIMU()
{
    if(!mCurrentFrame.mpPrevFrame) // 是否存在上一帧
    {
        Verbose::PrintMess("No last frame", Verbose::VERBOSITY_NORMAL);
        return false;
    }

    if(mbMapUpdated && mpLastKeyFrame) // 局部地图已更新且有上一关键帧
    {
        const Eigen::Vector3f twb1 = mpLastKeyFrame->GetImuPosition(); // 上一关键帧 IMU 位置
        const Eigen::Matrix3f Rwb1 = mpLastKeyFrame->GetImuRotation(); // 上一关键帧 IMU 旋转
        const Eigen::Vector3f Vwb1 = mpLastKeyFrame->GetVelocity();    // 上一关键帧 IMU 速度

        const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE); // 重力方向
        const float t12 = mpImuPreintegratedFromLastKF->dT;  // 从上一关键帧到当前帧的时间间隔

        // 预测旋转
        Eigen::Matrix3f Rwb2 = IMU::NormalizeRotation(Rwb1 * mpImuPreintegratedFromLastKF->GetDeltaRotation(mpLastKeyFrame->GetImuBias()));
        // 预测位置
        Eigen::Vector3f twb2 = twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1*mpImuPreintegratedFromLastKF->GetDeltaPosition(mpLastKeyFrame->GetImuBias());
        // 预测速度
        Eigen::Vector3f Vwb2 = Vwb1 + t12*Gz + Rwb1 * mpImuPreintegratedFromLastKF->GetDeltaVelocity(mpLastKeyFrame->GetImuBias());
        mCurrentFrame.SetImuPoseVelocity(Rwb2,twb2,Vwb2); // 设置当前帧 IMU 状态

        mCurrentFrame.mImuBias = mpLastKeyFrame->GetImuBias(); // 取上一关键帧的 IMU 偏置作为当前帧的 IMU 偏置
        mCurrentFrame.mPredBias = mCurrentFrame.mImuBias;      // 取当前帧的 IMU 偏置作为当前帧的 IMU 预测偏置
        return true;
    }
    else if(!mbMapUpdated) // 地图未更新
    {
        const Eigen::Vector3f twb1 = mLastFrame.GetImuPosition(); // 上一关键帧 IMU 位置
        const Eigen::Matrix3f Rwb1 = mLastFrame.GetImuRotation(); // 上一关键帧 IMU 旋转
        const Eigen::Vector3f Vwb1 = mLastFrame.GetVelocity();    // 上一关键帧 IMU 速度

        const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);         // 重力方向
        const float t12 = mCurrentFrame.mpImuPreintegratedFrame->dT; // 从上一帧到当前帧的时间间隔

        // 预测旋转
        Eigen::Matrix3f Rwb2 = IMU::NormalizeRotation(Rwb1 * mCurrentFrame.mpImuPreintegratedFrame->GetDeltaRotation(mLastFrame.mImuBias));
        // 预测位置
        Eigen::Vector3f twb2 = twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1 * mCurrentFrame.mpImuPreintegratedFrame->GetDeltaPosition(mLastFrame.mImuBias);
        // 预测速度
        Eigen::Vector3f Vwb2 = Vwb1 + t12*Gz + Rwb1 * mCurrentFrame.mpImuPreintegratedFrame->GetDeltaVelocity(mLastFrame.mImuBias);
        mCurrentFrame.SetImuPoseVelocity(Rwb2,twb2,Vwb2); // 设置当前帧 IMU 状态

        mCurrentFrame.mImuBias = mLastFrame.mImuBias;     // 取上一帧的 IMU 偏置作为当前帧的 IMU 偏置
        mCurrentFrame.mPredBias = mCurrentFrame.mImuBias; // 取当前帧的 IMU 偏置作为当前帧的 IMU 预测偏置
        return true;
    }
    else
        cout << "not IMU prediction!!" << endl;

    return false;
}

// 帧 IMU 重置（空实现）
void Tracking::ResetFrameIMU()
{
    // TODO To implement...
}



// 主跟踪函数，确定当前帧的位姿，筛选关键帧
void Tracking::Track()
{

    if (bStepByStep) // 单步调试
    {
        std::cout << "Tracking: Waiting to the next step" << std::endl;
        while(!mbStep && bStepByStep) // mbStep = true 执行一步，然后置为 false
            usleep(500);
        mbStep = false;
    }

    if(mpLocalMapper->mbBadImu) // 局部建图线程检测到 badImu，重置当前活跃地图
    {
        cout << "TRACK: Reset map because local mapper set the bad imu flag " << endl;
        mpSystem->ResetActiveMap();
        return;
    }

    Map* pCurrentMap = mpAtlas->GetCurrentMap(); // 获取当前地图
    if(!pCurrentMap)
    {
        cout << "ERROR: There is not an active map in the atlas" << endl;
    }

    if(mState!=NO_IMAGES_YET) // 时间戳检查
    {
        if(mLastFrame.mTimeStamp>mCurrentFrame.mTimeStamp) // 当前帧时间早于上一帧
        {
            cerr << "ERROR: Frame with a timestamp older than previous frame detected!" << endl;
            unique_lock<mutex> lock(mMutexImuQueue);
            mlQueueImuData.clear(); // 清空 IMU 队列
            CreateMapInAtlas();     // 在 Atlas 中创建新地图
            return;
        }
        else if(mCurrentFrame.mTimeStamp>mLastFrame.mTimeStamp+1.0) // 当前帧时间比上一帧大 1 秒以上
        {
            // cout << mCurrentFrame.mTimeStamp << ", " << mLastFrame.mTimeStamp << endl;
            // cout << "id last: " << mLastFrame.mnId << "    id curr: " << mCurrentFrame.mnId << endl;
            if(mpAtlas->isInertial()) // IMU 模式下
            {

                if(mpAtlas->isImuInitialized()) // IMU已初始化
                {
                    cout << "Timestamp jump detected. State set to LOST. Reseting IMU integration..." << endl;
                    if(!pCurrentMap->GetIniertialBA2()) // IMU 尚未完成第二次惯性 BA（IMU 刚完成初始化）
                    {
                        mpSystem->ResetActiveMap(); // 重置当前活跃地图
                    }
                    else // 已完成第二次惯性 BA
                    {
                        CreateMapInAtlas(); // 在 Atlas 中创建新地图
                    }
                }
                else // IMU 未初始化
                {
                    cout << "Timestamp jump detected, before IMU initialization. Reseting..." << endl;
                    mpSystem->ResetActiveMap(); // 重置当前活跃地图
                }
                return;
            }

        }
    }

    if ((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && mpLastKeyFrame)
        mCurrentFrame.SetNewBias(mpLastKeyFrame->GetImuBias()); // 将上一关键帧的 IMU 偏置传给当前帧

    if(mState==NO_IMAGES_YET) // 状态为无图像
    {
        mState = NOT_INITIALIZED; // 状态设为未初始化
    }

    mLastProcessedState = mState; // 上一次处理后的状态

    if ((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && !mbCreatedMap)
    {
#ifdef REGISTER_TIMES // 时间统计（REGISTER_TIMES 宏定义打开时，位于 include/Setting.h 中）
        std::chrono::steady_clock::time_point time_StartPreIMU = std::chrono::steady_clock::now();
#endif
        PreintegrateIMU(); // IMU 预积分（ IMU 模式且未创建地图）
#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_EndPreIMU = std::chrono::steady_clock::now();

        double timePreImu = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndPreIMU - time_StartPreIMU).count();
        vdIMUInteg_ms.push_back(timePreImu);
#endif

    }
    mbCreatedMap = false;

    // Get Map Mutex -> Map cannot be changed
    unique_lock<mutex> lock(pCurrentMap->mMutexMapUpdate); // 地图更新互斥锁

    mbMapUpdated = false;

    int nCurMapChangeIndex = pCurrentMap->GetMapChangeIndex(); // 当前地图更新索引
    int nMapChangeIndex = pCurrentMap->GetLastMapChange();     // 上一地图更新索引
    if(nCurMapChangeIndex>nMapChangeIndex) // 如果当前索引大于上次记录的索引，说明地图被局部建图线程更新过
    {
        pCurrentMap->SetLastMapChange(nCurMapChangeIndex); // 设上一地图更新索引为当前地图更新索引
        mbMapUpdated = true; // 地图更新设为 true
    }


    if(mState==NOT_INITIALIZED) // 当前状态为未初始化
    {
        if(mSensor==System::STEREO || mSensor==System::RGBD || mSensor==System::IMU_STEREO || mSensor==System::IMU_RGBD)
        {
            StereoInitialization(); // 双目/ RGB-D 初始化
        }
        else
        {
            MonocularInitialization(); // 单目初始化
        }

        //mpFrameDrawer->Update(this);

        if(mState!=OK) // 初始化后状态仍不是 OK  If rightly initialized, mState=OK
        {
            mLastFrame = Frame(mCurrentFrame); // 保存当前帧为上一帧
            return;
        }

        if(mpAtlas->GetAllMaps().size() == 1) // 初始化成功，且 Atlas 中只有一个地图
        {
            mnFirstFrameId = mCurrentFrame.mnId; // 记录当前帧 ID 作为第一帧 
        }
    }
    else // 已初始化
    {
        // System is initialized. Track Frame.
        bool bOK; // 位姿估计成功标志

#ifdef REGISTER_TIMES // 位姿预测时间统计（REGISTER_TIMES 宏定义打开时，位于 include/Setting.h 中）
        std::chrono::steady_clock::time_point time_StartPosePred = std::chrono::steady_clock::now();
#endif

        // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
        // 使用运动模型或重新定位（若跟踪丢失）进行初始相机姿态估计
        if(!mbOnlyTracking) // 普通模式
        {

            // State OK
            // Local Mapping is activated. This is the normal behaviour, unless
            // you explicitly activate the "only tracking" mode.
            if(mState==OK) // 状态为 OK
            {

                // Local Mapping might have changed some MapPoints tracked in last frame
                // 检查上一帧的地图点。局部建图线程可能会修改上一帧的一些跟踪地图点
                CheckReplacedInLastFrame();

                // 没有速度估计模型且 IMU 未初始化，或当前帧与上一重定位帧相差小于 2 帧
                if((!mbVelocity && !pCurrentMap->isImuInitialized()) || mCurrentFrame.mnId<mnLastRelocFrameId+2)
                {
                    Verbose::PrintMess("TRACK: Track with respect to the reference KF ", Verbose::VERBOSITY_DEBUG);
                    bOK = TrackReferenceKeyFrame(); // 基于参考关键帧进行位姿估计
                }
                else
                {
                    Verbose::PrintMess("TRACK: Track with motion model", Verbose::VERBOSITY_DEBUG);
                    bOK = TrackWithMotionModel(); // 基于运动模型进行位姿估计
                    if(!bOK) // 基于运动模型的跟踪失败，改为基于参考关键帧进行位姿估计
                        bOK = TrackReferenceKeyFrame();
                }


                if (!bOK) // 位姿估计失败
                {
                    // IMU 模式 且 最近有重定位，状态设为 LOST
                    if (mCurrentFrame.mnId<=(mnLastRelocFrameId+mnFramesToResetIMU) &&
                         (mSensor==System::IMU_MONOCULAR || mSensor==System::IMU_STEREO || mSensor == System::IMU_RGBD))
                    {
                        mState = LOST;
                    }
                    else if(pCurrentMap->KeyFramesInMap()>10) // 当前地图中关键帧数大于 10，状态设为 RECENTLY_LOST
                    {
                        // cout << "KF in map: " << pCurrentMap->KeyFramesInMap() << endl;
                        mState = RECENTLY_LOST;
                        mTimeStampLost = mCurrentFrame.mTimeStamp; // 记录丢失时间
                    }
                    else // 直接将状态设为 LOST
                    {
                        mState = LOST;
                    }
                }
            }
            else // 状态非 OK
            {

                if (mState == RECENTLY_LOST) // 状态为 RECENTLY_LOST
                {
                    Verbose::PrintMess("Lost for a short time", Verbose::VERBOSITY_NORMAL);

                    bOK = true;
                    if((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)) // IMU 模式
                    {
                        if(pCurrentMap->isImuInitialized()) // IMU 已初始化，则使用 IMU 预积分做位姿估计
                            PredictStateIMU();
                        else
                            bOK = false;

                        if (mCurrentFrame.mTimeStamp-mTimeStampLost>time_recently_lost) // 丢失时间超过 time_recently_lost（构造函数中初始化为 5.0s）
                        {
                            mState = LOST; // 状态设为 LOST
                            Verbose::PrintMess("Track Lost...", Verbose::VERBOSITY_NORMAL);
                            bOK=false;
                        }
                    }
                    else // 非 IMU 模式
                    {
                        // Relocalization
                        bOK = Relocalization(); // 尝试重定位
                        //std::cout << "mCurrentFrame.mTimeStamp:" << to_string(mCurrentFrame.mTimeStamp) << std::endl;
                        //std::cout << "mTimeStampLost:" << to_string(mTimeStampLost) << std::endl;
                        if(mCurrentFrame.mTimeStamp-mTimeStampLost>3.0f && !bOK) // 丢失时间超过 3s 且重定位失败
                        {
                            mState = LOST; // 状态设为 LOST
                            Verbose::PrintMess("Track Lost...", Verbose::VERBOSITY_NORMAL);
                            bOK=false;
                        }
                    }
                }
                else if (mState == LOST) // 状态为 LOST
                {

                    Verbose::PrintMess("A new map is started...", Verbose::VERBOSITY_NORMAL);

                    if (pCurrentMap->KeyFramesInMap()<10) // 当前活跃的地图中关键帧少于 10 个
                    {
                        mpSystem->ResetActiveMap(); // 重置当前活跃的地图
                        Verbose::PrintMess("Reseting current map...", Verbose::VERBOSITY_NORMAL);
                    }else
                        CreateMapInAtlas(); // 在 Atlas 中创建新地图

                    if(mpLastKeyFrame) // 确保上一关键帧清空
                        mpLastKeyFrame = static_cast<KeyFrame*>(NULL);

                    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

                    return;
                }
            }

        }
        else // 仅定位模式
        {
            // Localization Mode: Local Mapping is deactivated (TODO Not available in inertial mode)
            if(mState==LOST) // 状态为 LOST
            {
                if(mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
                    Verbose::PrintMess("IMU. State LOST", Verbose::VERBOSITY_NORMAL);
                bOK = Relocalization(); // 尝试重定位
            }
            else // 状态非 LOST
            {
                if(!mbVO) // 非 VO 模式
                {
                    // In last frame we tracked enough MapPoints in the map
                    if(mbVelocity) // 已有速度估计模型
                    {
                        bOK = TrackWithMotionModel(); // 基于运动模型位姿估计
                    }
                    else
                    {
                        bOK = TrackReferenceKeyFrame(); // 基于参考关键帧位姿估计
                    }
                }
                else // VO 模式
                {
                    // In last frame we tracked mainly "visual odometry" points.

                    // We compute two camera poses, one from motion model and one doing relocalization.
                    // If relocalization is sucessfull we choose that solution, otherwise we retain
                    // the "visual odometry" solution.

                    bool bOKMM = false;
                    bool bOKReloc = false;
                    vector<MapPoint*> vpMPsMM;
                    vector<bool> vbOutMM;
                    Sophus::SE3f TcwMM;
                    if(mbVelocity) // 已有速度估计
                    {
                        bOKMM = TrackWithMotionModel(); // 基于运动模型位姿估计
                        vpMPsMM = mCurrentFrame.mvpMapPoints;
                        vbOutMM = mCurrentFrame.mvbOutlier;
                        TcwMM = mCurrentFrame.GetPose();
                    }
                    bOKReloc = Relocalization(); // 尝试重定位

                    if(bOKMM && !bOKReloc) // 基于运动模型位姿估计和重定位都成功
                    {
                        // 优先考虑基于运动模型位姿估计的结果
                        mCurrentFrame.SetPose(TcwMM);
                        mCurrentFrame.mvpMapPoints = vpMPsMM;
                        mCurrentFrame.mvbOutlier = vbOutMM;

                        if(mbVO) // VO 模式
                        {
                            for(int i =0; i<mCurrentFrame.N; i++)
                            {
                                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                                {
                                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                                }
                            }
                        }
                    }
                    else if(bOKReloc) // 重定位成功
                    {
                        mbVO = false; // 退出 VO 模式
                    }

                    bOK = bOKReloc || bOKMM;
                }
            }
        }

        if(!mCurrentFrame.mpReferenceKF) // 当前帧没有参考关键帧，则关联跟踪线程维护的参考关键帧
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_EndPosePred = std::chrono::steady_clock::now();

        double timePosePred = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndPosePred - time_StartPosePred).count();
        vdPosePred_ms.push_back(timePosePred);
#endif


#ifdef REGISTER_TIMES // 局部地图跟踪时间统计（REGISTER_TIMES 宏定义打开时，位于 include/Setting.h 中）
        std::chrono::steady_clock::time_point time_StartLMTrack = std::chrono::steady_clock::now();
#endif
        // If we have an initial estimation of the camera pose and matching. Track the local map.
        // 如果我们对相机姿态和匹配有初始估计，就跟踪局部地图
        if(!mbOnlyTracking) // 普通模式
        {
            if(bOK) // 位姿估计成功
            {
                bOK = TrackLocalMap(); // 跟踪局部地图

            }
            if(!bOK) // 位姿估计失败
                cout << "Fail to track local map!" << endl;
        }
        else // 仅定位
        {
            // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
            // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
            // the camera we will use the local map again.
            if(bOK && !mbVO) // 位姿估计成功且非 VO 模式
                bOK = TrackLocalMap(); // 跟踪局部地图
        }

        if(bOK) // 局部地图跟踪成功，状态设为 OK 
            mState = OK;
        else if (mState == OK) // 局部地图跟踪失败，但状态为 OK（位姿估计成功）
        {
            if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) // IMU 模式
            {
                Verbose::PrintMess("Track lost for less than one second...", Verbose::VERBOSITY_NORMAL);
                if(!pCurrentMap->isImuInitialized() || !pCurrentMap->GetIniertialBA2()) // IMU 未初始化，或尚未完成第二次惯性 BA（IMU 刚完成初始化）
                {
                    cout << "IMU is not or recently initialized. Reseting active map..." << endl;
                    mpSystem->ResetActiveMap(); // 重置当前活跃地图
                }

                mState=RECENTLY_LOST; // 状态设为 RECENTLY_LOST
            }
            else // 非 IMU 模式，直接将状态设为 RECENTLY_LOST
            {
                mState=RECENTLY_LOST; // visual to lost
            }
            
            /*if(mCurrentFrame.mnId>mnLastRelocFrameId+mMaxFrames)
            {*/
            mTimeStampLost = mCurrentFrame.mTimeStamp; // 记录丢失时间
            //}
        }

        // Save frame if recent relocalization, since they are used for IMU reset (as we are making copy, it shluld be once mCurrFrame is completely modified)
        // 如果最近进行了重新定位，则保存当前帧，因为它们用于 IMU 重置（由于我们正在创建副本，因此应在 mCurrFrame 完全修改后才进行一次）
        if((mCurrentFrame.mnId<(mnLastRelocFrameId+mnFramesToResetIMU)) && (mCurrentFrame.mnId > mnFramesToResetIMU) &&
           (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && pCurrentMap->isImuInitialized())
        {
            // TODO check this situation
            Verbose::PrintMess("Saving pointer to frame. imu needs reset...", Verbose::VERBOSITY_NORMAL);
            Frame* pF = new Frame(mCurrentFrame);
            pF->mpPrevFrame = new Frame(mLastFrame);

            // Load preintegration
            pF->mpImuPreintegratedFrame = new IMU::Preintegrated(mCurrentFrame.mpImuPreintegratedFrame);
        }

        if(pCurrentMap->isImuInitialized()) // IMU 已初始化
        {
            if(bOK) // 局部地图跟踪成功
            {
                if(mCurrentFrame.mnId==(mnLastRelocFrameId+mnFramesToResetIMU))
                {
                    cout << "RESETING FRAME!!!" << endl;
                    ResetFrameIMU(); // 该函数实现为空
                }
                else if(mCurrentFrame.mnId>(mnLastRelocFrameId+30)) // 重定位超过 30 帧
                    mLastBias = mCurrentFrame.mImuBias; // 更新上一次 IMU 偏置
            }
        }

#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_EndLMTrack = std::chrono::steady_clock::now();

        double timeLMTrack = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndLMTrack - time_StartLMTrack).count();
        vdLMTrack_ms.push_back(timeLMTrack);
#endif

        // Update drawer
        mpFrameDrawer->Update(this); // 更新帧绘制器
        if(mCurrentFrame.isSet()) // 当前帧已设置位姿
            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.GetPose()); // 更新地图绘制器

        if(bOK || mState==RECENTLY_LOST) // 局部地图跟踪成功或状态为 RECENTLY_LOST
        {
            // 更新运动模型 Update motion model
            if(mLastFrame.isSet() && mCurrentFrame.isSet()) // 当前和上一帧已设置位姿
            {
                Sophus::SE3f LastTwc = mLastFrame.GetPose().inverse(); // 上一帧相对于世界的位姿
                mVelocity = mCurrentFrame.GetPose() * LastTwc; // 从上一帧到当前帧的相对运动
                mbVelocity = true;
            }
            else {
                mbVelocity = false;
            }

            if(mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) // IMU 模式
                mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.GetPose()); // 更新地图绘制器

            // 清除不可靠的 VO 匹配 Clean VO matches
            for(int i=0; i<mCurrentFrame.N; i++)
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(pMP)
                    if(pMP->Observations()<1)
                    {
                        mCurrentFrame.mvbOutlier[i] = false;
                        mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                    }
            }

            // 清除仅定位模式下由 UpdateLastFrame() 创建的临时地图点列表 Delete temporal MapPoints
            for(list<MapPoint*>::iterator lit = mlpTemporalPoints.begin(), lend =  mlpTemporalPoints.end(); lit!=lend; lit++)
            {
                MapPoint* pMP = *lit;
                delete pMP;
            }
            mlpTemporalPoints.clear();

#ifdef REGISTER_TIMES // 关键帧决策时间统计（REGISTER_TIMES 宏定义打开时，位于 include/Setting.h 中）
            std::chrono::steady_clock::time_point time_StartNewKF = std::chrono::steady_clock::now();
#endif
            bool bNeedKF = NeedNewKeyFrame(); // 判断是否需要插入新关键帧

            // Check if we need to insert a new keyframe
            // bNeedKF == true 且 局部地图跟踪成功（或 IMU 模式下 mInsertKFsLost == true 且 状态为 RECENTLY_LOST）
            if(bNeedKF && (bOK || (mInsertKFsLost && mState==RECENTLY_LOST && (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD))))
                CreateNewKeyFrame(); // 创建新的关键帧

#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_EndNewKF = std::chrono::steady_clock::now();

            double timeNewKF = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndNewKF - time_StartNewKF).count();
            vdNewKF_ms.push_back(timeNewKF);
#endif

            // We allow points with high innovation (considererd outliers by the Huber Function)
            // pass to the new keyframe, so that bundle adjustment will finally decide
            // if they are outliers or not. We don't want next frame to estimate its position
            // with those points so we discard them in the frame. Only has effect if lastframe is tracked
            for(int i=0; i<mCurrentFrame.N;i++) // 清理外点
            {
                if(mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                    mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
            }
        }

        // Reset if the camera get lost soon after initialization
        if(mState==LOST) // 处理 LOST 状态
        {
            if(pCurrentMap->KeyFramesInMap()<=10) // 当前地图关键帧不超过 10 个
            {
                mpSystem->ResetActiveMap(); // 重置当前活跃地图
                return;
            }
            if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) // IMU 模式下
                if (!pCurrentMap->isImuInitialized()) // IMU 未初始化
                {
                    Verbose::PrintMess("Track lost before IMU initialisation, reseting...", Verbose::VERBOSITY_QUIET);
                    mpSystem->ResetActiveMap(); // 重置当前活跃地图
                    return;
                }

            CreateMapInAtlas(); // 在 Atlas 中创建新地图

            return;
        }

        if(!mCurrentFrame.mpReferenceKF) // 当前帧没有参考关键帧，则关联跟踪线程维护的参考关键帧
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

        mLastFrame = Frame(mCurrentFrame); // 当前帧转为上一帧
    }

    if(mState==OK || mState==RECENTLY_LOST) // 状态为 OK 或 RECENTLY_LOST
    {
        // 保存位姿 Store frame pose information to retrieve the complete camera trajectory afterwards.
        if(mCurrentFrame.isSet()) // 当前帧已设置位姿
        {
            Sophus::SE3f Tcr_ = mCurrentFrame.GetPose() * mCurrentFrame.mpReferenceKF->GetPoseInverse(); // 参考帧相对于当前帧的位姿
            mlRelativeFramePoses.push_back(Tcr_);                 // 存放相对位姿
            mlpReferences.push_back(mCurrentFrame.mpReferenceKF); // 存放参考帧
            mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);     // 存放时间戳
            mlbLost.push_back(mState==LOST);                      // 此处 mState==LOST 恒为 false
        }
        else // 当前帧没有位姿
        {
            // This can happen if tracking is lost
            // 复制上一帧的信息
            mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
            mlpReferences.push_back(mlpReferences.back());
            mlFrameTimes.push_back(mlFrameTimes.back());
            mlbLost.push_back(mState==LOST);                      // 此处 mState==LOST 恒为 false
        }

    }

#ifdef REGISTER_LOOP
    if (Stop()) {

        // Safe area to stop
        while(isStopped())
        {
            usleep(3000);
        }
    }
#endif
}



// 双目/ RGB-D 初始化（包括创建初始地图）
void Tracking::StereoInitialization()
{
    if(mCurrentFrame.N>500) // 特征点数量大于 500 
    {
        if (mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) // 双目惯性或 RGB-D 惯性
        {
            if (!mCurrentFrame.mpImuPreintegrated || !mLastFrame.mpImuPreintegrated) // 上一关键帧到当前帧或上一帧的 IMU 预积分不存在
            {
                cout << "not IMU meas" << endl;
                return;
            }

            // 快速初始化标志 mFastInit == false 且 当前帧和上一帧的帧间 IMU 预积分中的平均加速度之差的范数小于 0.5
            if (!mFastInit && (mCurrentFrame.mpImuPreintegratedFrame->avgA-mLastFrame.mpImuPreintegratedFrame->avgA).norm()<0.5)
            {
                cout << "not enough acceleration" << endl;
                return;
            }

            // 重置上一关键帧到当前帧的预积分
            if(mpImuPreintegratedFromLastKF)
                delete mpImuPreintegratedFromLastKF;
            mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);
            mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;
        }

        // 设置初始位姿 Set Frame pose to the origin (In case of inertial SLAM to imu)
        if (mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) // 双目惯性或 RGB-D 惯性，工具IMU外参设置
        {
            Eigen::Matrix3f Rwb0 = mCurrentFrame.mImuCalib.mTcb.rotationMatrix();
            Eigen::Vector3f twb0 = mCurrentFrame.mImuCalib.mTcb.translation();
            Eigen::Vector3f Vwb0;
            Vwb0.setZero();
            mCurrentFrame.SetImuPoseVelocity(Rwb0, twb0, Vwb0);
        }
        else // 非双目惯性或 RGB-D 惯性，设为单位阵
            mCurrentFrame.SetPose(Sophus::SE3f());

        // 将当前帧作为初始关键帧 Create KeyFrame
        KeyFrame* pKFini = new KeyFrame(mCurrentFrame,mpAtlas->GetCurrentMap(),mpKeyFrameDB);

        // Insert KeyFrame in the map
        mpAtlas->AddKeyFrame(pKFini); // 将关键帧添加到地图中

        // 创建地图点并关联到关键帧 Create MapPoints and asscoiate to KeyFrame
        if(!mpCamera2){ // 普通双目和 RGB-D
            for(int i=0; i<mCurrentFrame.N;i++) // 遍历特征点
            {
                float z = mCurrentFrame.mvDepth[i]; // 特征点深度值
                if(z>0) // 特征点深度值有效
                {
                    Eigen::Vector3f x3D;
                    mCurrentFrame.UnprojectStereo(i, x3D); // 根据特征点像素坐标和深度，反投影得到相机坐标系下的三维点
                    MapPoint* pNewMP = new MapPoint(x3D, pKFini, mpAtlas->GetCurrentMap()); // 创建地图点
                    pNewMP->AddObservation(pKFini,i); // 给地图点添加观测：关键帧 pKFini 的第 i 个特征点
                    pKFini->AddMapPoint(pNewMP,i); // 给关键帧添加地图点关联：第 i 个特征点对应 pNewMP
                    pNewMP->ComputeDistinctiveDescriptors(); // 计算地图点的代表性描述子
                    pNewMP->UpdateNormalAndDepth(); // 更新地图点的法线和深度范围
                    mpAtlas->AddMapPoint(pNewMP); // 将地图点加入 Atlas 当前地图

                    mCurrentFrame.mvpMapPoints[i]=pNewMP; // 当前帧的第 i 个特征点关联到新地图点
                }
            }
        } 
        else{ // 鱼眼双目
            for(int i = 0; i < mCurrentFrame.Nleft; i++){ // 只遍历左目特征点
                int rightIndex = mCurrentFrame.mvLeftToRightMatch[i]; // 获取左目第 i 个特征点在右目中的匹配索引
                if(rightIndex != -1) // 只处理有右目匹配的点
                {
                    Eigen::Vector3f x3D = mCurrentFrame.mvStereo3Dpoints[i]; // 直接取计算好的三维点
                    MapPoint* pNewMP = new MapPoint(x3D, pKFini, mpAtlas->GetCurrentMap()); // 创建地图点

                    pNewMP->AddObservation(pKFini,i); // 左目观测
                    pNewMP->AddObservation(pKFini,rightIndex + mCurrentFrame.Nleft); // 右目观测
                    pKFini->AddMapPoint(pNewMP,i); // 左目关联
                    pKFini->AddMapPoint(pNewMP,rightIndex + mCurrentFrame.Nleft); // 右目关联
                    pNewMP->ComputeDistinctiveDescriptors(); // 计算地图点的代表性描述子
                    pNewMP->UpdateNormalAndDepth(); // 更新地图点的法线和深度范围
                    mpAtlas->AddMapPoint(pNewMP); // 将地图点加入 Atlas 当前地图

                    mCurrentFrame.mvpMapPoints[i]=pNewMP; // 左目特征点关联地图点
                    mCurrentFrame.mvpMapPoints[rightIndex + mCurrentFrame.Nleft]=pNewMP; // 右目特征点关联地图点
                }
            }
        }

        Verbose::PrintMess("New Map created with " + to_string(mpAtlas->MapPointsInMap()) + " points", Verbose::VERBOSITY_QUIET);

        //cout << "Active map: " << mpAtlas->GetCurrentMap()->GetId() << endl;

        mpLocalMapper->InsertKeyFrame(pKFini); // 将初始关键帧插入局部建图的队列

        mLastFrame = Frame(mCurrentFrame);     // 将当前帧复制为上一帧
        mnLastKeyFrameId = mCurrentFrame.mnId; // 将当前帧 ID 复制为上一帧 ID
        mpLastKeyFrame = pKFini;               // 将初始关键帧复制为上一关键帧
        //mnLastRelocFrameId = mCurrentFrame.mnId;

        mvpLocalKeyFrames.push_back(pKFini);          // 将初始关键帧加入局部关键帧列表
        mvpLocalMapPoints=mpAtlas->GetAllMapPoints(); // 局部地图点列表设为当前地图所有地图点
        mpReferenceKF = pKFini;                       // 设置跟踪线程的参考关键帧为初始关键帧
        mCurrentFrame.mpReferenceKF = pKFini;         // 当前帧的参考关键帧也设为初始关键帧

        mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints); // 设置局部地图点为 Atlas 的参考地图点

        mpAtlas->GetCurrentMap()->mvpKeyFrameOrigins.push_back(pKFini); // 将初始关键帧记录为地图的原点关键帧

        mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.GetPose()); // 更新地图绘制器的当前相机位姿

        mState=OK; // 状态设为 OK
    }
}

// 单目初始化
void Tracking::MonocularInitialization()
{

    if(!mbReadyToInitializate) // 未准备好初始化
    {
        // Set Reference Frame
        if(mCurrentFrame.mvKeys.size()>100) // 特征点数大于 100
        {

            mInitialFrame = Frame(mCurrentFrame); // 初始帧
            mLastFrame = Frame(mCurrentFrame);    // 上一帧
            // 初始帧特征点坐标
            mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
            for(size_t i=0; i<mCurrentFrame.mvKeysUn.size(); i++)
                mvbPrevMatched[i]=mCurrentFrame.mvKeysUn[i].pt;

            fill(mvIniMatches.begin(),mvIniMatches.end(),-1); // 初始化匹配索引（全部填充为 -1）

            if (mSensor == System::IMU_MONOCULAR) // 单目 IMU 模式
            {
                if(mpImuPreintegratedFromLastKF) // 重置上一关键帧到当前帧的预积分对象
                {
                    delete mpImuPreintegratedFromLastKF;
                }
                mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);
                mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;

            }

            mbReadyToInitializate = true; // 准备好进行初始化

            return;
        }
    }
    else // 准备好进行初始化
    {
        if (((int)mCurrentFrame.mvKeys.size()<=100)||((mSensor == System::IMU_MONOCULAR)&&(mLastFrame.mTimeStamp-mInitialFrame.mTimeStamp>1.0)))
        {
            // 当前帧特征点少于 100 或 单目惯性模式下初始帧和当前帧时间间隔超过 1s，回到寻找初始帧的 未准备好初始化 状态
            mbReadyToInitializate = false;
            return;
        }

        // Find correspondences
        ORBmatcher matcher(0.9,true); // 创建特征匹配器
        int nmatches = matcher.SearchForInitialization(mInitialFrame,mCurrentFrame,mvbPrevMatched,mvIniMatches,100); // 匹配当前帧和初始帧

        // Check if there are enough correspondences
        if(nmatches<100) // 匹配特征点小于 100 个
        {
            mbReadyToInitializate = false; // 回到寻找初始帧的 未准备好初始化 状态
            return;
        }

        Sophus::SE3f Tcw; // 世界相对于当前帧的位姿
        vector<bool> vbTriangulated; // 特征点三角化成功标志 Triangulated Correspondences (mvIniMatches)
        // 尝试两视图重建
        if(mpCamera->ReconstructWithTwoViews(mInitialFrame.mvKeysUn,mCurrentFrame.mvKeysUn,mvIniMatches,Tcw,mvIniP3D,vbTriangulated))
        {
            for(size_t i=0, iend=mvIniMatches.size(); i<iend;i++) // 标记未三角化的匹配
            {
                if(mvIniMatches[i]>=0 && !vbTriangulated[i])
                {
                    mvIniMatches[i]=-1;
                    nmatches--;
                }
            }

            // Set Frame Poses
            mInitialFrame.SetPose(Sophus::SE3f()); // 初始帧位姿设为单位阵
            mCurrentFrame.SetPose(Tcw);            // 当前帧位姿设为重建得到的 Tcw

            CreateInitialMapMonocular(); // 创建初始地图（单目）
        }
    }
}

// 创建初始地图（单目）
void Tracking::CreateInitialMapMonocular()
{
    // 将初始化用的两帧作为关键帧 Create KeyFrames
    KeyFrame* pKFini = new KeyFrame(mInitialFrame,mpAtlas->GetCurrentMap(),mpKeyFrameDB); // 初始关键帧（初始帧）
    KeyFrame* pKFcur = new KeyFrame(mCurrentFrame,mpAtlas->GetCurrentMap(),mpKeyFrameDB); // 当前关键帧（当前帧）

    if(mSensor == System::IMU_MONOCULAR) // 单目 IMU 模式
        pKFini->mpImuPreintegrated = (IMU::Preintegrated*)(NULL); // 初始关键帧预积分置空

    pKFini->ComputeBoW(); // 计算初始关键帧的词袋向量
    pKFcur->ComputeBoW(); // 计算当前关键帧的词袋向量

    // 在 Atlas 中加入关键帧 Insert KFs in the map
    mpAtlas->AddKeyFrame(pKFini);
    mpAtlas->AddKeyFrame(pKFcur);

    for(size_t i=0; i<mvIniMatches.size();i++) // 创建地图点
    {
        if(mvIniMatches[i]<0) // 跳过未三角化的匹配
            continue;

        //Create MapPoint.
        Eigen::Vector3f worldPos;
        worldPos << mvIniP3D[i].x, mvIniP3D[i].y, mvIniP3D[i].z; // 从三角化结果取出三维点
        MapPoint* pMP = new MapPoint(worldPos,pKFcur,mpAtlas->GetCurrentMap()); // 创建地图点

        pKFini->AddMapPoint(pMP,i); // 初始关键帧第 i 个特征点关联该地图点
        pKFcur->AddMapPoint(pMP,mvIniMatches[i]); // 当前关键帧第 mvIniMatches[i] 个特征点关联该地图点
        pMP->AddObservation(pKFini,i); // 地图点添加对初始关键帧的观测
        pMP->AddObservation(pKFcur,mvIniMatches[i]); // 地图点添加对当前关键帧的观测
        pMP->ComputeDistinctiveDescriptors(); // 计算地图点的代表性描述子
        pMP->UpdateNormalAndDepth(); // 更新地图点的法线和深度范围

        //Fill Current Frame structure
        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP; // 当前帧第 mvIniMatches[i] 个特征点关联地图点
        mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false; // 标记为非外点

        //Add to Map
        mpAtlas->AddMapPoint(pMP); // 将地图点加入 Atlas 当前地图
    }

    // 更新关键帧连接关系 Update Connections
    pKFini->UpdateConnections();
    pKFcur->UpdateConnections();

    std::set<MapPoint*> sMPs;
    sMPs = pKFini->GetMapPoints(); // 初始关键帧地图点

    // Bundle Adjustment
    Verbose::PrintMess("New Map created with " + to_string(mpAtlas->MapPointsInMap()) + " points", Verbose::VERBOSITY_QUIET);
    Optimizer::GlobalBundleAdjustemnt(mpAtlas->GetCurrentMap(),20); // 全局 BA，迭代 20 次，优化已有的两个关键帧位姿和地图点

    float medianDepth = pKFini->ComputeSceneMedianDepth(2); // 计算初始关键帧中地图点深度的中值
    float invMedianDepth; // medianDepth 倒数
    if(mSensor == System::IMU_MONOCULAR) // 单目 IMU 模式
        invMedianDepth = 4.0f/medianDepth; // 4.0f
    else
        invMedianDepth = 1.0f/medianDepth;

    // 检查初始化质量
    if(medianDepth<0 || pKFcur->TrackedMapPoints(1)<50) // TODO Check, originally 100 tracks
    {
        Verbose::PrintMess("Wrong initialization, reseting...", Verbose::VERBOSITY_QUIET);
        mpSystem->ResetActiveMap(); // 初始化质量差，重置当前地图
        return;
    }

    // 缩放当前帧位姿 Scale initial baseline
    Sophus::SE3f Tc2w = pKFcur->GetPose();
    Tc2w.translation() *= invMedianDepth;
    pKFcur->SetPose(Tc2w);

    // 缩放地图点 Scale points
    vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
    for(size_t iMP=0; iMP<vpAllMapPoints.size(); iMP++)
    {
        if(vpAllMapPoints[iMP])
        {
            MapPoint* pMP = vpAllMapPoints[iMP];
            pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth);
            pMP->UpdateNormalAndDepth();
        }
    }

    if (mSensor == System::IMU_MONOCULAR) // 单目 IMU
    {
        pKFcur->mPrevKF = pKFini; // 设当前关键帧的上一关键帧为初始关键帧
        pKFini->mNextKF = pKFcur; // 设初始关键帧的下一关键帧为当前关键帧
        pKFcur->mpImuPreintegrated = mpImuPreintegratedFromLastKF; // 记录上一关键帧到当前帧的 IMU 预积分对象

        mpImuPreintegratedFromLastKF = new IMU::Preintegrated(pKFcur->mpImuPreintegrated->GetUpdatedBias(),pKFcur->mImuCalib);
    }

    // 将关键帧插入局部建图线程的队列
    mpLocalMapper->InsertKeyFrame(pKFini);
    mpLocalMapper->InsertKeyFrame(pKFcur);
    mpLocalMapper->mFirstTs=pKFcur->mTimeStamp; // 记录初始关键帧的时间

    mCurrentFrame.SetPose(pKFcur->GetPose()); // 用全局 BA 的结果更新当前帧位姿
    mnLastKeyFrameId=mCurrentFrame.mnId;      // 取当前关键帧 ID 为上一关键帧 ID
    mpLastKeyFrame = pKFcur;                  // 取当前关键帧为上一关键帧
    //mnLastRelocFrameId = mInitialFrame.mnId;

    mvpLocalKeyFrames.push_back(pKFcur); // 将当前关键帧加入局部关键帧列表
    mvpLocalKeyFrames.push_back(pKFini); // 将初始关键帧加入局部关键帧列表
    mvpLocalMapPoints=mpAtlas->GetAllMapPoints(); // 将当前地图点加入局部地图点列表
    mpReferenceKF = pKFcur; // 设当前关键帧为跟踪线程参考帧
    mCurrentFrame.mpReferenceKF = pKFcur; // 当前帧的参考关键帧也设为当前关键帧

    // 计算初始速度 Compute here initial velocity
    vector<KeyFrame*> vKFs = mpAtlas->GetAllKeyFrames(); // 获取所有关键帧

    Sophus::SE3f deltaT = vKFs.back()->GetPose() * vKFs.front()->GetPoseInverse(); // 计算初始帧相对于当前帧的位姿
    mbVelocity = false;
    Eigen::Vector3f phi = deltaT.so3().log();

    double aux = (mCurrentFrame.mTimeStamp-mLastFrame.mTimeStamp)/(mCurrentFrame.mTimeStamp-mInitialFrame.mTimeStamp);
    phi *= aux;

    mLastFrame = Frame(mCurrentFrame); // 取当前帧为上一帧

    mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints); // 设局部地图点为参考地图点

    mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose()); // 更新地图绘制器

    mpAtlas->GetCurrentMap()->mvpKeyFrameOrigins.push_back(pKFini); // 将初始关键帧记录为地图的原点关键帧

    mState=OK; // 状态设为 OK

    initID = pKFcur->mnId; // 记录当前关键帧的 ID
}



// 在 Atlas 中创建地图
void Tracking::CreateMapInAtlas()
{
    mnLastInitFrameId = mCurrentFrame.mnId; // 取当前帧 ID 为上一次初始化帧 ID
    mpAtlas->CreateNewMap(); // 创建新地图
    if (mSensor==System::IMU_STEREO || mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_RGBD)
        mpAtlas->SetInertialSensor(); // IMU 模式下设置地图为惯性地图
    mbSetInit=false; // 初始化标志设为 false

    mnInitialFrameId = mCurrentFrame.mnId+1; // 设初始帧 ID 为当前帧的下一帧
    mState = NO_IMAGES_YET; // 状态设为 NO_IMAGES_YET

    // Restart the variable with information about the last KF
    mbVelocity = false; // 速度估计设为 false
    //mnLastRelocFrameId = mnLastInitFrameId; // The last relocation KF_id is the current id, because it is the new starting point for new map
    Verbose::PrintMess("First frame id in map: " + to_string(mnLastInitFrameId+1), Verbose::VERBOSITY_NORMAL);
    mbVO = false; // VO 模式设为 false， Init value for know if there are enough MapPoints in the last KF
    if(mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR)
    {
        mbReadyToInitializate = false; // 单目模式设为 未准备好初始化
    }

    if((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && mpImuPreintegratedFromLastKF)
    {
        // IMU 模式下重置预积分模型
        delete mpImuPreintegratedFromLastKF;
        mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);
    }

    if(mpLastKeyFrame) // 清空上一关键帧
        mpLastKeyFrame = static_cast<KeyFrame*>(NULL);

    if(mpReferenceKF) // 清空参考关键帧
        mpReferenceKF = static_cast<KeyFrame*>(NULL);

    mLastFrame = Frame();    // 重置上一帧
    mCurrentFrame = Frame(); // 重置当前帧
    mvIniMatches.clear();    // 清空初始化匹配

    mbCreatedMap = true; // 地图创建标志设为 true
}

// 检查上一帧中是否有地图点被替换
void Tracking::CheckReplacedInLastFrame()
{
    for(int i =0; i<mLastFrame.N; i++) // 遍历上一帧的特征点
    {
        MapPoint* pMP = mLastFrame.mvpMapPoints[i]; // 获取当前特征点关联的地图点

        if(pMP) // 该地图点是否存在
        {
            MapPoint* pRep = pMP->GetReplaced(); // 获取替换后的地图点
            if(pRep) // 存在替换点
            {
                mLastFrame.mvpMapPoints[i] = pRep; // 更新特征点关联的地图点
            }
        }
    }
}



// 基于参考关键帧进行位姿估计
// 先使用上一帧位姿作为初值，再匹配当前帧的特征点与参考关键帧的地图点，通过优化减小重投影误差完成当前帧位姿估计
bool Tracking::TrackReferenceKeyFrame()
{
    // Compute Bag of Words vector
    mCurrentFrame.ComputeBoW(); // 计算当前帧的词袋向量

    // We perform first an ORB matching with the reference keyframe
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.7,true); // 创建特征匹配器
    vector<MapPoint*> vpMapPointMatches; // 匹配结果容器

    int nmatches = matcher.SearchByBoW(mpReferenceKF,mCurrentFrame,vpMapPointMatches); // 使用词袋向量搜索匹配

    if(nmatches<15) // 匹配数少于 15，视为失败
    {
        cout << "TRACK_REF_KF: Less than 15 matches!!\n";
        return false;
    }

    mCurrentFrame.mvpMapPoints = vpMapPointMatches; // 将匹配结果赋给当前帧
    mCurrentFrame.SetPose(mLastFrame.GetPose()); // 用上一帧的位姿作为当前帧的初始位姿

    //mCurrentFrame.PrintPointDistribution();


    // cout << " TrackReferenceKeyFrame mLastFrame.mTcw:  " << mLastFrame.mTcw << endl;
    Optimizer::PoseOptimization(&mCurrentFrame); // 优化当前位姿

    // Discard outliers
    int nmatchesMap = 0; // 统计优化后仍然有效且被地图点观测的内点数量
    for(int i =0; i<mCurrentFrame.N; i++) // 遍历当前帧的特征点
    {
        //if(i >= mCurrentFrame.Nleft) break;
        if(mCurrentFrame.mvpMapPoints[i]) // 只处理有地图点匹配的特征点
        {
            if(mCurrentFrame.mvbOutlier[i]) // 处理外点
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i]; // 取出地图点

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL); // 将当前帧该特征点的地图点匹配置空
                mCurrentFrame.mvbOutlier[i]=false; // 复位外点标志
                if(i < mCurrentFrame.Nleft){ // 左目中该特征点的跟踪标志置为 false
                    pMP->mbTrackInView = false;
                }
                else{ // 右目中 该特征点的跟踪标志置为 false
                    pMP->mbTrackInViewR = false;
                }
                pMP->mbTrackInView = false; // 该特征点的跟踪标志置为 false
                pMP->mnLastFrameSeen = mCurrentFrame.mnId; // 记录最后看到该地图点的帧 ID
                nmatches--; // 总匹配数 -1
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0) // 统计有效地图点匹配
                nmatchesMap++;
        }
    }

    if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
        return true; // IMU 模式直接返回 true
    else
        return nmatchesMap>=10; // nmatchesMap>=10 时返回 true，否则返回 true
}

// 更新上一帧信息（位姿，临时地图点）
void Tracking::UpdateLastFrame()
{
    // Update pose according to reference keyframe
    KeyFrame* pRef = mLastFrame.mpReferenceKF; // 上一参考关键帧
    Sophus::SE3f Tlr = mlRelativeFramePoses.back(); // 参考关键帧相对于上一帧的位姿
    mLastFrame.SetPose(Tlr * pRef->GetPose()); // 恢复世界相对于上一帧的位姿

    if(mnLastKeyFrameId==mLastFrame.mnId || mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR || !mbOnlyTracking)
        return; // 上一帧是关键帧，或单目（+IMU）模式，或非仅定位模式，return 结束该函数

    // 创建 VO 模式的临时地图点 Create "visual odometry" MapPoints
    // We sort points according to their measured depth by the stereo/RGB-D sensor
    vector<pair<float,int> > vDepthIdx; // 深度-索引 对向量
    const int Nfeat = mLastFrame.Nleft == -1? mLastFrame.N : mLastFrame.Nleft; // 特征点数量
    vDepthIdx.reserve(Nfeat); // 预分配空间
    for(int i=0; i<Nfeat;i++) // 遍历特征点，收集深度值有效的
    {
        float z = mLastFrame.mvDepth[i];
        if(z>0)
        {
            vDepthIdx.push_back(make_pair(z,i));
        }
    }

    if(vDepthIdx.empty()) // 无有效深度值，return 结束该函数
        return;

    sort(vDepthIdx.begin(),vDepthIdx.end()); // 按深度值排序

    // We insert all close points (depth<mThDepth)
    // If less than 100 close points, we insert the 100 closest ones.
    int nPoints = 0; // 计数
    for(size_t j=0; j<vDepthIdx.size();j++) // 遍历排序后的点
    {
        int i = vDepthIdx[j].second;

        bool bCreateNew = false; // 初始化已创建标志

        MapPoint* pMP = mLastFrame.mvpMapPoints[i]; // 获取当前特征点关联的地图点

        if(!pMP) // 无关联地图点
            bCreateNew = true;
        else if(pMP->Observations()<1) // 关联地图点没有被观测到
            bCreateNew = true;

        if(bCreateNew) // 创建临时地图点
        {
            Eigen::Vector3f x3D;

            if(mLastFrame.Nleft == -1){
                mLastFrame.UnprojectStereo(i, x3D);
            }
            else{
                x3D = mLastFrame.UnprojectStereoFishEye(i);
            }

            MapPoint* pNewMP = new MapPoint(x3D,mpAtlas->GetCurrentMap(),&mLastFrame,i);
            mLastFrame.mvpMapPoints[i]=pNewMP;

            mlpTemporalPoints.push_back(pNewMP);
            nPoints++;
        }
        else // 已有可靠地图点
        {
            nPoints++;
        }

        if(vDepthIdx[j].first>mThDepth && nPoints>100) // 当前点的深度超过阈值且处理超过 100 个点，提前结束循环
            break;
    }
}

// 基于运动模型进行位姿估计
// IMU 可用时，直接用 IMU 预积分完成位姿估计
// IMU 不可用时，先使用 mVelocity 计算当前帧位姿初值，再匹配当前帧的特征点与上一键帧中被跟踪的地图点，通过优化减小重投影误差完成当前帧位姿估计
bool Tracking::TrackWithMotionModel()
{
    ORBmatcher matcher(0.9,true); // 创建特征匹配器

    // Update last frame pose according to its reference keyframe
    // Create "visual odometry" points if in Localization Mode
    UpdateLastFrame(); // 更新最上一帧

    if (mpAtlas->isImuInitialized() && (mCurrentFrame.mnId>mnLastRelocFrameId+mnFramesToResetIMU))
    {
        // Predict state with IMU if it is initialized and it doesnt need reset
        PredictStateIMU(); // IMU 已初始化且不需要重置时，使用 IMU 预积分做位姿估计
        return true;
    }
    else
    {
        mCurrentFrame.SetPose(mVelocity * mLastFrame.GetPose()); // 用恒速模型 mVelocity（ 局部跟踪成功后或状态为 RECENTLY_LOST 会更新） 估计位姿
    }

    fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL)); // 清空当前帧地图点匹配

    // Project points seen in previous frame
    int th; // 投影匹配时的搜索半径

    if(mSensor==System::STEREO) // 双目
        th=7;
    else // 其它
        th=15;

    int nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,th,mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR); // 投影匹配

    // If few matches, uses a wider window search
    if(nmatches<20) // 匹配少于 20，增大 th 值再次尝试
    {
        Verbose::PrintMess("Not enough matches, wider window search!!", Verbose::VERBOSITY_NORMAL);
        fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));

        nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,2*th,mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR);
        Verbose::PrintMess("Matches with wider search: " + to_string(nmatches), Verbose::VERBOSITY_NORMAL);

    }

    if(nmatches<20) // 增大 th 值后匹配仍少于 20
    {
        Verbose::PrintMess("Not enough matches!!", Verbose::VERBOSITY_NORMAL);
        if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
            return true; // IMU 模式返回 true
        else
            return false; // 非 IMU 模式返回 false
    }

    // Optimize frame pose with all matches
    Optimizer::PoseOptimization(&mCurrentFrame); // 优化当前位姿

    // Discard outliers
    int nmatchesMap = 0; // 统计优化后仍然有效且被地图点观测的内点数量
    for(int i =0; i<mCurrentFrame.N; i++) // 遍历当前帧的特征点
    {
        if(mCurrentFrame.mvpMapPoints[i]) // 只处理有地图点匹配的特征点
        {
            if(mCurrentFrame.mvbOutlier[i]) // 处理外点
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i]; // 取出地图点

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL); // 将当前帧该特征点的地图点匹配置空
                mCurrentFrame.mvbOutlier[i]=false; // 复位外点标志
                if(i < mCurrentFrame.Nleft){ // 左目中该特征点的跟踪标志置为 false
                    pMP->mbTrackInView = false;
                }
                else{ // 右目中 该特征点的跟踪标志置为 false
                    pMP->mbTrackInViewR = false;
                }
                pMP->mnLastFrameSeen = mCurrentFrame.mnId; // 记录最后看到该地图点的帧 ID
                nmatches--; // 总匹配数 -1
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0) // 统计有效地图点匹配
                nmatchesMap++;
        }
    }

    if(mbOnlyTracking) // 仅定位模式
    {
        mbVO = nmatchesMap<10; // 有效地图点匹配少于 10，则进入 VO 模式
        return nmatches>20; // 匹配数大于 20，返回 true
    }

    if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
        return true; // IMU 模式直接返回 true
    else
        return nmatchesMap>=10; // 非 IMU 模式下，有效地图点匹配大于等于 10，则返回 true
}



// 局部地图跟踪，根据局部地图优化当前帧位姿
bool Tracking::TrackLocalMap()
{

    // We have an estimation of the camera pose and some map points tracked in the frame.
    // We retrieve the local map and try to find matches to points in the local map.
    // 我们估计了相机姿态，并在图像中跟踪了一些地图点。接下来将检索局部地图，尝试与局部地图中的点进行匹配。
    mTrackedFr++; // 记录成功进入局部地图跟踪的帧数

    UpdateLocalMap(); // 更新局部地图
    SearchLocalPoints(); // 搜索局部地图点

    // TOO check outliers before PO
    int aux1 = 0, aux2=0;
    for(int i=0; i<mCurrentFrame.N; i++)
        if( mCurrentFrame.mvpMapPoints[i])
        {
            aux1++; // 优化前有地图点匹配的特征点数
            if(mCurrentFrame.mvbOutlier[i])
                aux2++; // 优化前被标记为外点的数量
        }

    int inliers;
    if (!mpAtlas->isImuInitialized()) // IMU 未初始化
        Optimizer::PoseOptimization(&mCurrentFrame); // 纯视觉位姿优化
    else
    {
        if(mCurrentFrame.mnId<=mnLastRelocFrameId+mnFramesToResetIMU) // 刚发生重定位，IMU 状态尚不稳定
        {
            Verbose::PrintMess("TLM: PoseOptimization ", Verbose::VERBOSITY_DEBUG);
            Optimizer::PoseOptimization(&mCurrentFrame); // 纯视觉位姿优化
        }
        else
        {
            // if(!mbMapUpdated && mState == OK) //  && (mnMatchesInliers>30))
            if(!mbMapUpdated) //  && (mnMatchesInliers>30)) // 局部地图未更新（函数在进行位姿估计前会检测）
            {
                Verbose::PrintMess("TLM: PoseInertialOptimizationLastFrame ", Verbose::VERBOSITY_DEBUG);
                // 以上一帧的 IMU 状态为参考，融合视觉和 IMU 约束优化当前帧位姿
                inliers = Optimizer::PoseInertialOptimizationLastFrame(&mCurrentFrame); // , !mpLastKeyFrame->GetMap()->GetIniertialBA1());
            }
            else // 局部地图已更新
            {
                Verbose::PrintMess("TLM: PoseInertialOptimizationLastKeyFrame ", Verbose::VERBOSITY_DEBUG);
                // 以上一关键帧的 IMU 状态为参考，融合视觉和 IMU 约束优化当前帧位姿
                inliers = Optimizer::PoseInertialOptimizationLastKeyFrame(&mCurrentFrame); // , !mpLastKeyFrame->GetMap()->GetIniertialBA1());
            }
        }
    }

    aux1 = 0, aux2 = 0;
    for(int i=0; i<mCurrentFrame.N; i++)
        if( mCurrentFrame.mvpMapPoints[i])
        {
            aux1++; // 优化后仍有地图点匹配的特征点数
            if(mCurrentFrame.mvbOutlier[i])
                aux2++; // 优化后仍被标记为外点的数量
        }

    mnMatchesInliers = 0; // 匹配内点计数

    // Update MapPoints Statistics
    for(int i=0; i<mCurrentFrame.N; i++) // 遍历当前帧所有特征点
    {
        if(mCurrentFrame.mvpMapPoints[i]) // 第 i 个特征点有地图点匹配
        {
            if(!mCurrentFrame.mvbOutlier[i]) // 不是外点
            {
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound(); // 增加地图点的“被找到”计数
                if(!mbOnlyTracking) // 非仅定位模式
                {
                    if(mCurrentFrame.mvpMapPoints[i]->Observations()>0) // 至少一个关键帧观测到该地图点
                        mnMatchesInliers++; // 内点数 +1
                }
                else // 仅定位模式
                    mnMatchesInliers++; // 内点数 +1
            }
            else if(mSensor==System::STEREO) // 外点且是双目
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL); // 直接将当前帧的该地图点匹配置空
        }
    }

    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently
    mpLocalMapper->mnMatchesInliers=mnMatchesInliers; // 将当前帧的匹配内点数传给局部建图线程
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && mnMatchesInliers<50) // 当前帧 ID < 最后重定位帧 ID + FPS 且 匹配内点数小于 50
        return false;

    if((mnMatchesInliers>10)&&(mState==RECENTLY_LOST)) // 匹配内点数大于 10 且 状态为 RECENTLY_LOST
        return true;


    if (mSensor == System::IMU_MONOCULAR) // 单目 IMU 模式
    {
        if((mnMatchesInliers<15 && mpAtlas->isImuInitialized())||(mnMatchesInliers<50 && !mpAtlas->isImuInitialized()))
        {
            return false; // （匹配内点数少于 15 且 IMU 已初始化） 或 （匹配内点数少于 15 且 IMU 未初始化）
        }
        else
            return true;
    }
    else if (mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) // 双目/RGB-D IMU 模式
    {
        if(mnMatchesInliers<15)
        {
            return false; // 匹配内点数少于 15
        }
        else
            return true;
    }
    else // 纯相机
    {
        if(mnMatchesInliers<30)
            return false; // 匹配内点数少于 30
        else
            return true;
    }
}



// 关键帧决策
bool Tracking::NeedNewKeyFrame()
{
    if((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && !mpAtlas->GetCurrentMap()->isImuInitialized())
    {
        if (mSensor == System::IMU_MONOCULAR && (mCurrentFrame.mTimeStamp-mpLastKeyFrame->mTimeStamp)>=0.25)
            return true;
        else if ((mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && (mCurrentFrame.mTimeStamp-mpLastKeyFrame->mTimeStamp)>=0.25)
            return true;
        else
            return false;
    }

    if(mbOnlyTracking)
        return false;

    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested()) {
        /*if(mSensor == System::MONOCULAR)
        {
            std::cout << "NeedNewKeyFrame: localmap stopped" << std::endl;
        }*/
        return false;
    }

    const int nKFs = mpAtlas->KeyFramesInMap();

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && nKFs>mMaxFrames)
    {
        return false;
    }

    // Tracked MapPoints in the reference keyframe
    int nMinObs = 3;
    if(nKFs<=2)
        nMinObs=2;
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

    // Local Mapping accept keyframes?
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

    // Check how many "close" points are being tracked and how many could be potentially created.
    int nNonTrackedClose = 0;
    int nTrackedClose= 0;

    if(mSensor!=System::MONOCULAR && mSensor!=System::IMU_MONOCULAR)
    {
        int N = (mCurrentFrame.Nleft == -1) ? mCurrentFrame.N : mCurrentFrame.Nleft;
        for(int i =0; i<N; i++)
        {
            if(mCurrentFrame.mvDepth[i]>0 && mCurrentFrame.mvDepth[i]<mThDepth)
            {
                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                    nTrackedClose++;
                else
                    nNonTrackedClose++;

            }
        }
        //Verbose::PrintMess("[NEEDNEWKF]-> closed points: " + to_string(nTrackedClose) + "; non tracked closed points: " + to_string(nNonTrackedClose), Verbose::VERBOSITY_NORMAL);// Verbose::VERBOSITY_DEBUG);
    }

    bool bNeedToInsertClose;
    bNeedToInsertClose = (nTrackedClose<100) && (nNonTrackedClose>70);

    // Thresholds
    float thRefRatio = 0.75f;
    if(nKFs<2)
        thRefRatio = 0.4f;

    /*int nClosedPoints = nTrackedClose + nNonTrackedClose;
    const int thStereoClosedPoints = 15;
    if(nClosedPoints < thStereoClosedPoints && (mSensor==System::STEREO || mSensor==System::IMU_STEREO))
    {
        //Pseudo-monocular, there are not enough close points to be confident about the stereo observations.
        thRefRatio = 0.9f;
    }*/

    if(mSensor==System::MONOCULAR)
        thRefRatio = 0.9f;

    if(mpCamera2) thRefRatio = 0.75f;

    if(mSensor==System::IMU_MONOCULAR)
    {
        if(mnMatchesInliers>350) // Points tracked from the local map
            thRefRatio = 0.75f;
        else
            thRefRatio = 0.90f;
    }

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    const bool c1a = mCurrentFrame.mnId>=mnLastKeyFrameId+mMaxFrames;
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    const bool c1b = ((mCurrentFrame.mnId>=mnLastKeyFrameId+mMinFrames) && bLocalMappingIdle); //mpLocalMapper->KeyframesInQueue() < 2);
    //Condition 1c: tracking is weak
    const bool c1c = mSensor!=System::MONOCULAR && mSensor!=System::IMU_MONOCULAR && mSensor!=System::IMU_STEREO && mSensor!=System::IMU_RGBD && (mnMatchesInliers<nRefMatches*0.25 || bNeedToInsertClose) ;
    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    const bool c2 = (((mnMatchesInliers<nRefMatches*thRefRatio || bNeedToInsertClose)) && mnMatchesInliers>15);

    //std::cout << "NeedNewKF: c1a=" << c1a << "; c1b=" << c1b << "; c1c=" << c1c << "; c2=" << c2 << std::endl;
    // Temporal condition for Inertial cases
    bool c3 = false;
    if(mpLastKeyFrame)
    {
        if (mSensor==System::IMU_MONOCULAR)
        {
            if ((mCurrentFrame.mTimeStamp-mpLastKeyFrame->mTimeStamp)>=0.5)
                c3 = true;
        }
        else if (mSensor==System::IMU_STEREO || mSensor == System::IMU_RGBD)
        {
            if ((mCurrentFrame.mTimeStamp-mpLastKeyFrame->mTimeStamp)>=0.5)
                c3 = true;
        }
    }

    bool c4 = false;
    if ((((mnMatchesInliers<75) && (mnMatchesInliers>15)) || mState==RECENTLY_LOST) && (mSensor == System::IMU_MONOCULAR)) // MODIFICATION_2, originally ((((mnMatchesInliers<75) && (mnMatchesInliers>15)) || mState==RECENTLY_LOST) && ((mSensor == System::IMU_MONOCULAR)))
        c4=true;
    else
        c4=false;

    if(((c1a||c1b||c1c) && c2)||c3 ||c4)
    {
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise send a signal to interrupt BA
        if(bLocalMappingIdle || mpLocalMapper->IsInitializing())
        {
            return true;
        }
        else
        {
            mpLocalMapper->InterruptBA();
            if(mSensor!=System::MONOCULAR  && mSensor!=System::IMU_MONOCULAR)
            {
                if(mpLocalMapper->KeyframesInQueue()<3)
                    return true;
                else
                    return false;
            }
            else
            {
                //std::cout << "NeedNewKeyFrame: localmap is busy" << std::endl;
                return false;
            }
        }
    }
    else
        return false;
}

// 创建新的关键帧
void Tracking::CreateNewKeyFrame()
{
    if(mpLocalMapper->IsInitializing() && !mpAtlas->isImuInitialized())
        return;

    if(!mpLocalMapper->SetNotStop(true))
        return;

    KeyFrame* pKF = new KeyFrame(mCurrentFrame,mpAtlas->GetCurrentMap(),mpKeyFrameDB);

    if(mpAtlas->isImuInitialized()) //  || mpLocalMapper->IsInitializing())
        pKF->bImu = true;

    pKF->SetNewBias(mCurrentFrame.mImuBias);
    mpReferenceKF = pKF;
    mCurrentFrame.mpReferenceKF = pKF;

    if(mpLastKeyFrame)
    {
        pKF->mPrevKF = mpLastKeyFrame;
        mpLastKeyFrame->mNextKF = pKF;
    }
    else
        Verbose::PrintMess("No last KF in KF creation!!", Verbose::VERBOSITY_NORMAL);

    // Reset preintegration from last KF (Create new object)
    if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
    {
        mpImuPreintegratedFromLastKF = new IMU::Preintegrated(pKF->GetImuBias(),pKF->mImuCalib);
    }

    if(mSensor!=System::MONOCULAR && mSensor != System::IMU_MONOCULAR) // TODO check if incluide imu_stereo
    {
        mCurrentFrame.UpdatePoseMatrices();
        // cout << "create new MPs" << endl;
        // We sort points by the measured depth by the stereo/RGBD sensor.
        // We create all those MapPoints whose depth < mThDepth.
        // If there are less than 100 close points we create the 100 closest.
        int maxPoint = 100;
        if(mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
            maxPoint = 100;

        vector<pair<float,int> > vDepthIdx;
        int N = (mCurrentFrame.Nleft != -1) ? mCurrentFrame.Nleft : mCurrentFrame.N;
        vDepthIdx.reserve(mCurrentFrame.N);
        for(int i=0; i<N; i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                vDepthIdx.push_back(make_pair(z,i));
            }
        }

        if(!vDepthIdx.empty())
        {
            sort(vDepthIdx.begin(),vDepthIdx.end());

            int nPoints = 0;
            for(size_t j=0; j<vDepthIdx.size();j++)
            {
                int i = vDepthIdx[j].second;

                bool bCreateNew = false;

                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(!pMP)
                    bCreateNew = true;
                else if(pMP->Observations()<1)
                {
                    bCreateNew = true;
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
                }

                if(bCreateNew)
                {
                    Eigen::Vector3f x3D;

                    if(mCurrentFrame.Nleft == -1){
                        mCurrentFrame.UnprojectStereo(i, x3D);
                    }
                    else{
                        x3D = mCurrentFrame.UnprojectStereoFishEye(i);
                    }

                    MapPoint* pNewMP = new MapPoint(x3D,pKF,mpAtlas->GetCurrentMap());
                    pNewMP->AddObservation(pKF,i);

                    //Check if it is a stereo observation in order to not
                    //duplicate mappoints
                    if(mCurrentFrame.Nleft != -1 && mCurrentFrame.mvLeftToRightMatch[i] >= 0){
                        mCurrentFrame.mvpMapPoints[mCurrentFrame.Nleft + mCurrentFrame.mvLeftToRightMatch[i]]=pNewMP;
                        pNewMP->AddObservation(pKF,mCurrentFrame.Nleft + mCurrentFrame.mvLeftToRightMatch[i]);
                        pKF->AddMapPoint(pNewMP,mCurrentFrame.Nleft + mCurrentFrame.mvLeftToRightMatch[i]);
                    }

                    pKF->AddMapPoint(pNewMP,i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpAtlas->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;
                    nPoints++;
                }
                else
                {
                    nPoints++;
                }

                if(vDepthIdx[j].first>mThDepth && nPoints>maxPoint)
                {
                    break;
                }
            }
            //Verbose::PrintMess("new mps for stereo KF: " + to_string(nPoints), Verbose::VERBOSITY_NORMAL);
        }
    }


    mpLocalMapper->InsertKeyFrame(pKF);

    mpLocalMapper->SetNotStop(false);

    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKF;
}



// 在当前帧中搜索局部地图点
void Tracking::SearchLocalPoints()
{
    // Do not search map points already matched
    for(vector<MapPoint*>::iterator vit=mCurrentFrame.mvpMapPoints.begin(), vend=mCurrentFrame.mvpMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP)
        {
            if(pMP->isBad())
            {
                *vit = static_cast<MapPoint*>(NULL);
            }
            else
            {
                pMP->IncreaseVisible();
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                pMP->mbTrackInView = false;
                pMP->mbTrackInViewR = false;
            }
        }
    }

    int nToMatch=0;

    // Project points in frame and check its visibility
    for(vector<MapPoint*>::iterator vit=mvpLocalMapPoints.begin(), vend=mvpLocalMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;

        if(pMP->mnLastFrameSeen == mCurrentFrame.mnId)
            continue;
        if(pMP->isBad())
            continue;
        // Project (this fills MapPoint variables for matching)
        if(mCurrentFrame.isInFrustum(pMP,0.5))
        {
            pMP->IncreaseVisible();
            nToMatch++;
        }
        if(pMP->mbTrackInView)
        {
            mCurrentFrame.mmProjectPoints[pMP->mnId] = cv::Point2f(pMP->mTrackProjX, pMP->mTrackProjY);
        }
    }

    if(nToMatch>0)
    {
        ORBmatcher matcher(0.8);
        int th = 1;
        if(mSensor==System::RGBD || mSensor==System::IMU_RGBD)
            th=3;
        if(mpAtlas->isImuInitialized())
        {
            if(mpAtlas->GetCurrentMap()->GetIniertialBA2())
                th=2;
            else
                th=6;
        }
        else if(!mpAtlas->isImuInitialized() && (mSensor==System::IMU_MONOCULAR || mSensor==System::IMU_STEREO || mSensor == System::IMU_RGBD))
        {
            th=10;
        }

        // If the camera has been relocalised recently, perform a coarser search
        if(mCurrentFrame.mnId<mnLastRelocFrameId+2)
            th=5;

        if(mState==LOST || mState==RECENTLY_LOST) // Lost for less than 1 second
            th=15; // 15

        int matches = matcher.SearchByProjection(mCurrentFrame, mvpLocalMapPoints, th, mpLocalMapper->mbFarPoints, mpLocalMapper->mThFarPoints);
    }
}

// 更新局部地图
void Tracking::UpdateLocalMap()
{
    // This is for visualization
    mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints); // 设局部地图点为参考地图点

    // Update
    UpdateLocalKeyFrames(); // 更新局部关键帧
    UpdateLocalPoints(); // 更新局部地图点
}

// 更新局部地图点，确定局部地图点
void Tracking::UpdateLocalPoints()
{
    mvpLocalMapPoints.clear(); // 清空局部地图点

    int count_pts = 0; // 地图点计数

    // 逆序遍历局部关键帧
    for(vector<KeyFrame*>::const_reverse_iterator itKF=mvpLocalKeyFrames.rbegin(), itEndKF=mvpLocalKeyFrames.rend(); itKF!=itEndKF; ++itKF)
    {
        KeyFrame* pKF = *itKF;
        const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches(); // 该关键帧中每个特征点对应的地图点匹配

        for(vector<MapPoint*>::const_iterator itMP=vpMPs.begin(), itEndMP=vpMPs.end(); itMP!=itEndMP; itMP++) // 遍历地图点
        {

            MapPoint* pMP = *itMP;
            if(!pMP) // 空匹配
                continue;
            if(pMP->mnTrackReferenceForFrame==mCurrentFrame.mnId) // 跳过已被当前帧处理过的点
                continue;
            if(!pMP->isBad()) // 处理非坏点
            {
                count_pts++; // 地图点 +1
                mvpLocalMapPoints.push_back(pMP); // 将该地图点加入局部地图点
                pMP->mnTrackReferenceForFrame=mCurrentFrame.mnId; // 设置该地图点的 mnTrackReferenceForFrame 为当前帧 ID，即该地图点已记录
            }
        }
    }
}

// 更新局部关键帧，确定跟踪线程参考关键帧和当前帧参考关键帧
void Tracking::UpdateLocalKeyFrames()
{
    // Each map point vote for the keyframes in which it has been observed
    // 每个地图点会为它所观测到的关键帧投票
    map<KeyFrame*,int> keyframeCounter; // 投票记录
    if(!mpAtlas->isImuInitialized() || (mCurrentFrame.mnId<mnLastRelocFrameId+2)) // IMU 为初始化 或 刚重定位不久（ 2 内）
    {
        for(int i=0; i<mCurrentFrame.N; i++) // 使用当前帧做统计
        {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i]; // 取第 i 个特征点的地图点匹配
            if(pMP) // 处理有匹配的点
            {
                if(!pMP->isBad()) // 处理非坏点
                {
                    const map<KeyFrame*,tuple<int,int>> observations = pMP->GetObservations(); // 返回该地图点的所有观测
                    for(map<KeyFrame*,tuple<int,int>>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                        keyframeCounter[it->first]++; // 遍历所有观测，对每个关键帧 it->first 的投票计数加一
                }
                else // 处理坏点
                {
                    mCurrentFrame.mvpMapPoints[i]=NULL; // 将当前帧的该匹配置空
                }
            }
        }
    }
    else
    {
        for(int i=0; i<mLastFrame.N; i++) // 使用上一帧做统计
        {
            // Using lastframe since current frame has not matches yet
            if(mLastFrame.mvpMapPoints[i])
            {
                MapPoint* pMP = mLastFrame.mvpMapPoints[i];
                if(!pMP)
                    continue;
                if(!pMP->isBad())
                {
                    const map<KeyFrame*,tuple<int,int>> observations = pMP->GetObservations();
                    for(map<KeyFrame*,tuple<int,int>>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                        keyframeCounter[it->first]++;
                }
                else
                {
                    // MODIFICATION
                    mLastFrame.mvpMapPoints[i]=NULL;
                }
            }
        }
    }


    int max=0; // 最多票数
    KeyFrame* pKFmax= static_cast<KeyFrame*>(NULL); // 票数最多的关键帧

    mvpLocalKeyFrames.clear(); // 清空局部关键帧
    mvpLocalKeyFrames.reserve(3*keyframeCounter.size()); // 预分配内存

    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    // 所有观察到地图点的关键帧都会被包含在局部地图中。同时检查哪些键帧共享最多的点
    for(map<KeyFrame*,int>::const_iterator it=keyframeCounter.begin(), itEnd=keyframeCounter.end(); it!=itEnd; it++) // 遍历投票记录
    {
        KeyFrame* pKF = it->first; // 取关键帧

        if(pKF->isBad()) // 跳过坏关键帧
            continue;

        if(it->second>max) // 找出票数最多的关键帧
        {
            max=it->second;
            pKFmax=pKF;
        }

        mvpLocalKeyFrames.push_back(pKF); // 将该关键帧加入局部关键帧
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId; // 标记该关键帧已被当前帧引用，避免后续重复加入
    }

    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    // 也包含一些尚未包含的、与已包含的关键帧相邻的关键帧
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++) // 遍历局部关键帧
    {
        // Limit the number of keyframes
        if(mvpLocalKeyFrames.size()>80) // 限制最多 80 个关键帧
            break;

        KeyFrame* pKF = *itKF;

        const vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10); // 取与 pKF 共视程度最高的前 10 个关键帧


        for(vector<KeyFrame*>::const_iterator itNeighKF=vNeighs.begin(), itEndNeighKF=vNeighs.end(); itNeighKF!=itEndNeighKF; itNeighKF++) // 遍历共视关键帧
        {
            KeyFrame* pNeighKF = *itNeighKF;
            if(!pNeighKF->isBad()) // 跳过质量不好的
            {
                if(pNeighKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId) // 如果还没有被当前帧引用
                {
                    mvpLocalKeyFrames.push_back(pNeighKF); // 加入局部关键帧
                    pNeighKF->mnTrackReferenceForFrame=mCurrentFrame.mnId; // 标记引用
                    break; // 每个 pKF 的共视关键帧只取一个加入局部关键帧
                }
            }
        }

        const set<KeyFrame*> spChilds = pKF->GetChilds(); // 返回 pKF 的子关键帧集合
        for(set<KeyFrame*>::const_iterator sit=spChilds.begin(), send=spChilds.end(); sit!=send; sit++) // 遍历子关键帧集合
        {
            KeyFrame* pChildKF = *sit;
            if(!pChildKF->isBad()) // 跳过质量不好的
            {
                if(pChildKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId) // 如果还没有被当前帧引用
                {
                    mvpLocalKeyFrames.push_back(pChildKF); // 加入局部关键帧
                    pChildKF->mnTrackReferenceForFrame=mCurrentFrame.mnId; // 标记引用
                    break; // 每个 pKF 的子关键帧只取一个加入局部关键帧
                }
            }
        }

        KeyFrame* pParent = pKF->GetParent(); // 返回 pKF 的父关键帧
        if(pParent) // 存在父关键帧
        {
            if(pParent->mnTrackReferenceForFrame!=mCurrentFrame.mnId) // 父关键帧还未被引用
            {
                mvpLocalKeyFrames.push_back(pParent); // 加入局部关键帧
                pParent->mnTrackReferenceForFrame=mCurrentFrame.mnId; // 标记应用
                break; // 结束局部关键帧遍历
            }
        }
    }

    // 添加10个最新的时间相关KF（主要用于IMU） Add 10 last temporal KFs (mainly for IMU)
    // IMU 模式 且 局部关键帧数量小于 80
    if((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) &&mvpLocalKeyFrames.size()<80)
    {
        KeyFrame* tempKeyFrame = mCurrentFrame.mpLastKeyFrame; // 从当前帧的上一关键帧开始

        const int Nd = 20; // 最多加入 20 个最近关键帧
        for(int i=0; i<Nd; i++){
            if (!tempKeyFrame) // 如果 tempKeyFrame 为空，跳出
                break;
            if(tempKeyFrame->mnTrackReferenceForFrame!=mCurrentFrame.mnId) // 该关键帧还未被引用
            {
                mvpLocalKeyFrames.push_back(tempKeyFrame); // 加入局部关键帧
                tempKeyFrame->mnTrackReferenceForFrame=mCurrentFrame.mnId; // 标记引用
                tempKeyFrame=tempKeyFrame->mPrevKF; // 沿 mPrevKF 链向前回溯
            }
        }
    }

    if(pKFmax) // 找到了投票最多的关键帧
    {
        mpReferenceKF = pKFmax; // 更新跟踪线程的参考关键帧
        mCurrentFrame.mpReferenceKF = mpReferenceKF; // 更新当前帧的参考关键帧
    }
}



// 重定位
bool Tracking::Relocalization()
{
    Verbose::PrintMess("Starting relocalization", Verbose::VERBOSITY_NORMAL);
    // Compute Bag of Words Vector
    mCurrentFrame.ComputeBoW();

    // Relocalization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame, mpAtlas->GetCurrentMap());

    if(vpCandidateKFs.empty()) {
        Verbose::PrintMess("There are not candidates", Verbose::VERBOSITY_NORMAL);
        return false;
    }

    const int nKFs = vpCandidateKFs.size();

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.75,true);

    vector<MLPnPsolver*> vpMLPnPsolvers;
    vpMLPnPsolvers.resize(nKFs);

    vector<vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    int nCandidates=0;

    for(int i=0; i<nKFs; i++)
    {
        KeyFrame* pKF = vpCandidateKFs[i];
        if(pKF->isBad())
            vbDiscarded[i] = true;
        else
        {
            int nmatches = matcher.SearchByBoW(pKF,mCurrentFrame,vvpMapPointMatches[i]);
            if(nmatches<15)
            {
                vbDiscarded[i] = true;
                continue;
            }
            else
            {
                MLPnPsolver* pSolver = new MLPnPsolver(mCurrentFrame,vvpMapPointMatches[i]);
                pSolver->SetRansacParameters(0.99,10,300,6,0.5,5.991);  //This solver needs at least 6 points
                vpMLPnPsolvers[i] = pSolver;
                nCandidates++;
            }
        }
    }

    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    bool bMatch = false;
    ORBmatcher matcher2(0.9,true);

    while(nCandidates>0 && !bMatch)
    {
        for(int i=0; i<nKFs; i++)
        {
            if(vbDiscarded[i])
                continue;

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            MLPnPsolver* pSolver = vpMLPnPsolvers[i];
            Eigen::Matrix4f eigTcw;
            bool bTcw = pSolver->iterate(5,bNoMore,vbInliers,nInliers, eigTcw);

            // If Ransac reachs max. iterations discard keyframe
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--;
            }

            // If a Camera Pose is computed, optimize
            if(bTcw)
            {
                Sophus::SE3f Tcw(eigTcw);
                mCurrentFrame.SetPose(Tcw);
                // Tcw.copyTo(mCurrentFrame.mTcw);

                set<MapPoint*> sFound;

                const int np = vbInliers.size();

                for(int j=0; j<np; j++)
                {
                    if(vbInliers[j])
                    {
                        mCurrentFrame.mvpMapPoints[j]=vvpMapPointMatches[i][j];
                        sFound.insert(vvpMapPointMatches[i][j]);
                    }
                    else
                        mCurrentFrame.mvpMapPoints[j]=NULL;
                }

                int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                if(nGood<10)
                    continue;

                for(int io =0; io<mCurrentFrame.N; io++)
                    if(mCurrentFrame.mvbOutlier[io])
                        mCurrentFrame.mvpMapPoints[io]=static_cast<MapPoint*>(NULL);

                // If few inliers, search by projection in a coarse window and optimize again
                if(nGood<50)
                {
                    int nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,10,100);

                    if(nadditional+nGood>=50)
                    {
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        if(nGood>30 && nGood<50)
                        {
                            sFound.clear();
                            for(int ip =0; ip<mCurrentFrame.N; ip++)
                                if(mCurrentFrame.mvpMapPoints[ip])
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                            nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,3,64);

                            // Final optimization
                            if(nGood+nadditional>=50)
                            {
                                nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                                for(int io =0; io<mCurrentFrame.N; io++)
                                    if(mCurrentFrame.mvbOutlier[io])
                                        mCurrentFrame.mvpMapPoints[io]=NULL;
                            }
                        }
                    }
                }


                // If the pose is supported by enough inliers stop ransacs and continue
                if(nGood>=50)
                {
                    bMatch = true;
                    break;
                }
            }
        }
    }

    if(!bMatch)
    {
        return false;
    }
    else
    {
        mnLastRelocFrameId = mCurrentFrame.mnId;
        cout << "Relocalized!!" << endl;
        return true;
    }

}

// 重置系统
void Tracking::Reset(bool bLocMap)
{
    Verbose::PrintMess("System Reseting", Verbose::VERBOSITY_NORMAL);

    if(mpViewer)
    {
        mpViewer->RequestStop();
        while(!mpViewer->isStopped())
            usleep(3000);
    }

    // Reset Local Mapping
    if (!bLocMap)
    {
        Verbose::PrintMess("Reseting Local Mapper...", Verbose::VERBOSITY_NORMAL);
        mpLocalMapper->RequestReset();
        Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);
    }


    // Reset Loop Closing
    Verbose::PrintMess("Reseting Loop Closing...", Verbose::VERBOSITY_NORMAL);
    mpLoopClosing->RequestReset();
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear BoW Database
    Verbose::PrintMess("Reseting Database...", Verbose::VERBOSITY_NORMAL);
    mpKeyFrameDB->clear();
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear Map (this erase MapPoints and KeyFrames)
    mpAtlas->clearAtlas();
    mpAtlas->CreateNewMap();
    if (mSensor==System::IMU_STEREO || mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_RGBD)
        mpAtlas->SetInertialSensor();
    mnInitialFrameId = 0;

    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    mState = NO_IMAGES_YET;

    mbReadyToInitializate = false;
    mbSetInit=false;

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();
    mCurrentFrame = Frame();
    mnLastRelocFrameId = 0;
    mLastFrame = Frame();
    mpReferenceKF = static_cast<KeyFrame*>(NULL);
    mpLastKeyFrame = static_cast<KeyFrame*>(NULL);
    mvIniMatches.clear();

    if(mpViewer)
        mpViewer->Release();

    Verbose::PrintMess("   End reseting! ", Verbose::VERBOSITY_NORMAL);
}

// 重置当前活跃地图
void Tracking::ResetActiveMap(bool bLocMap)
{
    Verbose::PrintMess("Active map Reseting", Verbose::VERBOSITY_NORMAL);
    if(mpViewer)
    {
        mpViewer->RequestStop();
        while(!mpViewer->isStopped())
            usleep(3000);
    }

    Map* pMap = mpAtlas->GetCurrentMap();

    if (!bLocMap)
    {
        Verbose::PrintMess("Reseting Local Mapper...", Verbose::VERBOSITY_VERY_VERBOSE);
        mpLocalMapper->RequestResetActiveMap(pMap);
        Verbose::PrintMess("done", Verbose::VERBOSITY_VERY_VERBOSE);
    }

    // Reset Loop Closing
    Verbose::PrintMess("Reseting Loop Closing...", Verbose::VERBOSITY_NORMAL);
    mpLoopClosing->RequestResetActiveMap(pMap);
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear BoW Database
    Verbose::PrintMess("Reseting Database", Verbose::VERBOSITY_NORMAL);
    mpKeyFrameDB->clearMap(pMap); // Only clear the active map references
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear Map (this erase MapPoints and KeyFrames)
    mpAtlas->clearMap();


    //KeyFrame::nNextId = mpAtlas->GetLastInitKFid();
    //Frame::nNextId = mnLastInitFrameId;
    mnLastInitFrameId = Frame::nNextId;
    //mnLastRelocFrameId = mnLastInitFrameId;
    mState = NO_IMAGES_YET; //NOT_INITIALIZED;

    mbReadyToInitializate = false;

    list<bool> lbLost;
    // lbLost.reserve(mlbLost.size());
    unsigned int index = mnFirstFrameId;
    cout << "mnFirstFrameId = " << mnFirstFrameId << endl;
    for(Map* pMap : mpAtlas->GetAllMaps())
    {
        if(pMap->GetAllKeyFrames().size() > 0)
        {
            if(index > pMap->GetLowerKFID())
                index = pMap->GetLowerKFID();
        }
    }

    //cout << "First Frame id: " << index << endl;
    int num_lost = 0;
    cout << "mnInitialFrameId = " << mnInitialFrameId << endl;

    for(list<bool>::iterator ilbL = mlbLost.begin(); ilbL != mlbLost.end(); ilbL++)
    {
        if(index < mnInitialFrameId)
            lbLost.push_back(*ilbL);
        else
        {
            lbLost.push_back(true);
            num_lost += 1;
        }

        index++;
    }
    cout << num_lost << " Frames set to lost" << endl;

    mlbLost = lbLost;

    mnInitialFrameId = mCurrentFrame.mnId;
    mnLastRelocFrameId = mCurrentFrame.mnId;

    mCurrentFrame = Frame();
    mLastFrame = Frame();
    mpReferenceKF = static_cast<KeyFrame*>(NULL);
    mpLastKeyFrame = static_cast<KeyFrame*>(NULL);
    mvIniMatches.clear();

    mbVelocity = false;

    if(mpViewer)
        mpViewer->Release();

    Verbose::PrintMess("   End reseting! ", Verbose::VERBOSITY_NORMAL);
}

// 返回当前局部地图点列表
vector<MapPoint*> Tracking::GetLocalMapMPS()
{
    return mvpLocalMapPoints;
}

// 更改相机标定？？？
void Tracking::ChangeCalibration(const string &strSettingPath)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    mK_.setIdentity();
    mK_(0,0) = fx;
    mK_(1,1) = fy;
    mK_(0,2) = cx;
    mK_(1,2) = cy;

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    Frame::mbInitialComputations = true;
}

// 仅定位
void Tracking::InformOnlyTracking(const bool &flag)
{
    mbOnlyTracking = flag;
}

// 更新帧 IMU？？？
void Tracking::UpdateFrameIMU(const float s, const IMU::Bias &b, KeyFrame* pCurrentKeyFrame)
{
    Map * pMap = pCurrentKeyFrame->GetMap();
    unsigned int index = mnFirstFrameId;
    list<ORB_SLAM3::KeyFrame*>::iterator lRit = mlpReferences.begin();
    list<bool>::iterator lbL = mlbLost.begin();
    for(auto lit=mlRelativeFramePoses.begin(),lend=mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lbL++)
    {
        if(*lbL)
            continue;

        KeyFrame* pKF = *lRit;

        while(pKF->isBad())
        {
            pKF = pKF->GetParent();
        }

        if(pKF->GetMap() == pMap)
        {
            (*lit).translation() *= s;
        }
    }

    mLastBias = b;

    mpLastKeyFrame = pCurrentKeyFrame;

    mLastFrame.SetNewBias(mLastBias);
    mCurrentFrame.SetNewBias(mLastBias);

    while(!mCurrentFrame.imuIsPreintegrated())
    {
        usleep(500);
    }


    if(mLastFrame.mnId == mLastFrame.mpLastKeyFrame->mnFrameId)
    {
        mLastFrame.SetImuPoseVelocity(mLastFrame.mpLastKeyFrame->GetImuRotation(),
                                      mLastFrame.mpLastKeyFrame->GetImuPosition(),
                                      mLastFrame.mpLastKeyFrame->GetVelocity());
    }
    else
    {
        const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);
        const Eigen::Vector3f twb1 = mLastFrame.mpLastKeyFrame->GetImuPosition();
        const Eigen::Matrix3f Rwb1 = mLastFrame.mpLastKeyFrame->GetImuRotation();
        const Eigen::Vector3f Vwb1 = mLastFrame.mpLastKeyFrame->GetVelocity();
        float t12 = mLastFrame.mpImuPreintegrated->dT;

        mLastFrame.SetImuPoseVelocity(IMU::NormalizeRotation(Rwb1*mLastFrame.mpImuPreintegrated->GetUpdatedDeltaRotation()),
                                      twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1*mLastFrame.mpImuPreintegrated->GetUpdatedDeltaPosition(),
                                      Vwb1 + Gz*t12 + Rwb1*mLastFrame.mpImuPreintegrated->GetUpdatedDeltaVelocity());
    }

    if (mCurrentFrame.mpImuPreintegrated)
    {
        const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);

        const Eigen::Vector3f twb1 = mCurrentFrame.mpLastKeyFrame->GetImuPosition();
        const Eigen::Matrix3f Rwb1 = mCurrentFrame.mpLastKeyFrame->GetImuRotation();
        const Eigen::Vector3f Vwb1 = mCurrentFrame.mpLastKeyFrame->GetVelocity();
        float t12 = mCurrentFrame.mpImuPreintegrated->dT;

        mCurrentFrame.SetImuPoseVelocity(IMU::NormalizeRotation(Rwb1*mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaRotation()),
                                      twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1*mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaPosition(),
                                      Vwb1 + Gz*t12 + Rwb1*mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaVelocity());
    }

    mnFirstImuFrameId = mCurrentFrame.mnId;
}

// 新建数据集
void Tracking::NewDataset()
{
    mnNumDataset++;
}

// 返回当前数据集编号
int Tracking::GetNumberDataset()
{
    return mnNumDataset;
}

// 返回当前帧匹配点数量
int Tracking::GetMatchesInliers()
{
    return mnMatchesInliers;
}

// 保存子轨迹
void Tracking::SaveSubTrajectory(string strNameFile_frames, string strNameFile_kf, string strFolder)
{
    mpSystem->SaveTrajectoryEuRoC(strFolder + strNameFile_frames);
    //mpSystem->SaveKeyFrameTrajectoryEuRoC(strFolder + strNameFile_kf);
}

// 保存指定地图的子轨迹
void Tracking::SaveSubTrajectory(string strNameFile_frames, string strNameFile_kf, Map* pMap)
{
    mpSystem->SaveTrajectoryEuRoC(strNameFile_frames, pMap);
    if(!strNameFile_kf.empty())
        mpSystem->SaveKeyFrameTrajectoryEuRoC(strNameFile_kf, pMap);
}

// 返回图像缩放系数
float Tracking::GetImageScale()
{
    return mImageScale;
}

#ifdef REGISTER_LOOP
void Tracking::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopRequested = true;
}

bool Tracking::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    if(mbStopRequested && !mbNotStop)
    {
        mbStopped = true;
        cout << "Tracking STOP" << endl;
        return true;
    }

    return false;
}

bool Tracking::stopRequested()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopRequested;
}

bool Tracking::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

void Tracking::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopped = false;
    mbStopRequested = false;
}
#endif

} //namespace ORB_SLAM
