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

#include<iostream>    // 标准输入输出
#include<algorithm>   // 算法
#include<fstream>     // 文件操作
#include<chrono>      // C++11 时间
#include <ctime>      // C 时间
#include <sstream>    // 字符串

#include<opencv2/core/core.hpp>  // opencv 模块

#include<System.h>         // slam 算法
#include "ImuTypes.h"      // IMU 数据类型定义

using namespace std;

void LoadImagesTUMVI(const string &strPathLeft, const string &strPathRight, const string &strPathTimes,
                vector<string> &vstrImageLeft, vector<string> &vstrImageRight, vector<double> &vTimeStamps);

void LoadIMU(const string &strImuPath, vector<double> &vTimeStamps, vector<cv::Point3f> &vAcc, vector<cv::Point3f> &vGyro);

double ttrack_tot = 0; // 跟踪总耗时
int main(int argc, char **argv) // argc 命令行参数个数，argv 参数字符串数组
{
    
    if(argc < 7) // 参数个数小于7，参数不够
    {
        // 可执行文件、词典文件、配置文件
        // 序列 1 左相机图像文件夹路径、右相机图像文件夹路径、图像时间戳文件路径、IMU 数据文件路径
        // 序列 2 ... 序列 n
        // 轨迹文件名（可选）
        cerr << endl << "Usage: ./stereo_inertial_tum_vi path_to_vocabulary path_to_settings "
                        "path_to_image_folder_1 path_to_image_folder_2 path_to_times_file path_to_imu_data "
                        "(trajectory_file_name)" << endl;
        return 1;
    }

    const int num_seq = (argc-3)/4; // 计算序列个数
    cout << "num_seq = " << num_seq << endl;
    bool bFileName= (((argc-3) % 4) == 1); // 参数中是否含有轨迹文件名
    string file_name;
    if (bFileName)
        file_name = string(argv[argc-1]); // 轨迹文件名


    // Load all sequences:
    int seq;
    vector< vector<string> > vstrImageLeftFilenames;    // 每个序列的左图文件名列表
    vector< vector<string> > vstrImageRightFilenames;   // 每个序列的右图文件名列表
    vector< vector<double> > vTimestampsCam;            // 每个序列的图像时间戳
    vector< vector<cv::Point3f> > vAcc, vGyro;          // 每个序列的 IMU 数据
    vector< vector<double> > vTimestampsImu;            // 每个序列的 IMU 数据时间戳
    vector<int> nImages;                                // 每个序列的图像数量
    vector<int> nImu;                                   // 每个序列的 IMU 数据数量
    vector<int> first_imu(num_seq,0);                   // 每个序列中与第一张图像时间对齐的 IMU 数据索引，初始化为 0

    // 根据序列数量调整外层 vector 大小
    vstrImageLeftFilenames.resize(num_seq);
    vstrImageRightFilenames.resize(num_seq);
    vTimestampsCam.resize(num_seq);
    vAcc.resize(num_seq);
    vGyro.resize(num_seq);
    vTimestampsImu.resize(num_seq);
    nImages.resize(num_seq);
    nImu.resize(num_seq);

    int tot_images = 0; // 累计所有序列的图像总数
    for (seq = 0; seq<num_seq; seq++) // 遍历每个序列
    {
        // 读取图像数据
        cout << "Loading images for sequence " << seq << "...";
        LoadImagesTUMVI(string(argv[4*(seq+1)-1]), string(argv[4*(seq+1)]), string(argv[4*(seq+1)+1]),
                         vstrImageLeftFilenames[seq], vstrImageRightFilenames[seq], vTimestampsCam[seq]);
        cout << "Total images: " << vstrImageLeftFilenames[seq].size() << endl;  // 图像数量
        cout << "Total cam ts: " << vTimestampsCam[seq].size() << endl;          // 相机时间戳数量
        cout << "first cam ts: " << vTimestampsCam[seq][0] << endl;              // 第一个相机时间戳
        cout << "LOADED!" << endl;

        // 读取 IMU 数据
        cout << "Loading IMU for sequence " << seq << "...";
        LoadIMU(string(argv[4*(seq+1)+2]), vTimestampsImu[seq], vAcc[seq], vGyro[seq]);
        cout << "Total IMU meas: " << vTimestampsImu[seq].size() << endl;  // IMU 数量
        cout << "first IMU ts: " << vTimestampsImu[seq][0] << endl;        // IMU 时间戳数量
        cout << "LOADED!" << endl;

        // 记录每个序列的图像数量和 IMU 数据数量，以及累计总图像数量
        nImages[seq] = vstrImageLeftFilenames[seq].size();
        tot_images += nImages[seq];
        nImu[seq] = vTimestampsImu[seq].size();

        // 图像或 IMU 数据为空则报错退出
        if((nImages[seq]<=0)||(nImu[seq]<=0))
        {
            cerr << "ERROR: Failed to load images or IMU for sequence" << seq << endl;
            return 1;
        }

        // Find first imu to be considered, supposing imu measurements start first
        // 找到与第一张图像时间戳对齐的 IMU 数据索引
        while(vTimestampsImu[seq][first_imu[seq]]<=vTimestampsCam[seq][0])
            first_imu[seq]++;
        first_imu[seq]--; // first imu measurement to be considered

    }

    // Vector for tracking time statistics
    // 跟踪耗时统计容器
    vector<float> vTimesTrack;
    vTimesTrack.resize(tot_images);

    cout << endl << "-------" << endl;
    cout.precision(17);  // 设置浮点输出精度为 17 位

    /*cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl;
    cout << "IMU data in the sequence: " << nImu << endl << endl;*/

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(argv[1],argv[2],ORB_SLAM3::System::IMU_STEREO, true, 0, file_name); // 创建系统
    float imageScale = SLAM.GetImageScale();  // 图像缩放系数，来源于 argv[2] 的配置文件？？？

    double t_resize = 0.f;  // 图像缩放耗时
    double t_track = 0.f;   // 跟踪耗时

    int proccIm = 0; // 记录处理的图像数量
    for (seq = 0; seq<num_seq; seq++) // 逐个序列处理
    {

        // Main loop
        cv::Mat imLeft, imRight;
        vector<ORB_SLAM3::IMU::Point> vImuMeas; // 两帧之间的 IMU 数据容器
        proccIm = 0;
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8)); // 图像增强？？？

        for(int ni=0; ni<nImages[seq]; ni++, proccIm++) // 逐张图像处理
        {

            // Read image from file
            // 以灰度图模式读取左右图像
            imLeft = cv::imread(vstrImageLeftFilenames[seq][ni],cv::IMREAD_GRAYSCALE);
            imRight = cv::imread(vstrImageRightFilenames[seq][ni],cv::IMREAD_GRAYSCALE);

            if(imLeft.empty() || imRight.empty()) // 图像为空，报错退出
            {
                cerr << endl << "Failed to load image at: "
                     <<  vstrImageLeftFilenames[seq][ni] << endl;
                return 1;
            }

            if(imageScale != 1.f) // imageScale 不为 1.0，则执行图像缩放
            {
#ifdef REGISTER_TIMES // 记录图像缩放耗时（REGISTER_TIMES 宏定义打开时，位于 include/Setting.h 中）
    #ifdef COMPILEDWITHC11
                std::chrono::steady_clock::time_point t_Start_Resize = std::chrono::steady_clock::now();
    #else
                std::chrono::monotonic_clock::time_point t_Start_Resize = std::chrono::monotonic_clock::now();
    #endif
#endif
                // 计算缩放后的尺寸并执行缩放
                int width = imLeft.cols * imageScale;
                int height = imLeft.rows * imageScale;
                cv::resize(imLeft, imLeft, cv::Size(width, height));
                cv::resize(imRight, imRight, cv::Size(width, height));
#ifdef REGISTER_TIMES // 结束图像缩放耗时记录
    #ifdef COMPILEDWITHC11
                std::chrono::steady_clock::time_point t_End_Resize = std::chrono::steady_clock::now();
    #else
                std::chrono::monotonic_clock::time_point t_End_Resize = std::chrono::monotonic_clock::now();
    #endif
                t_resize = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t_End_Resize - t_Start_Resize).count();
                SLAM.InsertResizeTime(t_resize);
#endif
            }

            // 对左右图像应用 clahe
            clahe->apply(imLeft,imLeft);
            clahe->apply(imRight,imRight);

            double tframe = vTimestampsCam[seq][ni]; // 当前图像时间戳

            // Load imu measurements from previous frame
            vImuMeas.clear(); // 清空两帧之间的 IMU 数据容器

            if(ni>0) // 首张图象跳过
            {
                // cout << "t_cam " << tframe << endl;

                while(vTimestampsImu[seq][first_imu[seq]]<=vTimestampsCam[seq][ni]) // 截取两帧之间的 IMU 数据
                {
                    // vImuMeas.push_back(ORB_SLAM3::IMU::Point(vAcc[first_imu],vGyro[first_imu],vTimestampsImu[first_imu]));
                    vImuMeas.push_back(ORB_SLAM3::IMU::Point(vAcc[seq][first_imu[seq]].x,vAcc[seq][first_imu[seq]].y,vAcc[seq][first_imu[seq]].z,
                                                             vGyro[seq][first_imu[seq]].x,vGyro[seq][first_imu[seq]].y,vGyro[seq][first_imu[seq]].z,
                                                             vTimestampsImu[seq][first_imu[seq]]));
                    // cout << "t_imu = " << fixed << vImuMeas.back().t << endl;
                    first_imu[seq]++; // IMU 数据索引 +1
                }
            }

            /*cout << "first imu: " << first_imu[seq] << endl;
            cout << "first imu time: " << fixed << vTimestampsImu[seq][0] << endl;
            cout << "size vImu: " << vImuMeas.size() << endl;*/

    #ifdef COMPILEDWITHC11 // 记录跟踪耗时
            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    #else
            std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
    #endif

            // Pass the image to the SLAM system
            SLAM.TrackStereo(imLeft,imRight,tframe,vImuMeas); // 执行跟踪

    #ifdef COMPILEDWITHC11 // 结束跟踪耗时记录
            std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    #else
            std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
    #endif

#ifdef REGISTER_TIMES // 记录实际跟踪（resize + tracking）耗时（REGISTER_TIMES 宏定义打开时，位于 include/Setting.h 中）
            t_track = t_resize + std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t2 - t1).count();
            SLAM.InsertTrackTime(t_track);
#endif

            double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count(); // 当前帧跟踪耗时
            ttrack_tot += ttrack; // 跟踪总耗时
            // std::cout << "ttrack: " << ttrack << std::endl;
            vTimesTrack[ni]=ttrack; // 存放当前帧跟踪耗时，多序列时会被下一序列覆盖？？？

            // Wait to load the next frame
            double T=0;
            if(ni<nImages[seq]-1) // 当前不为最后一帧，计算与下一帧的时间间隔
                T = vTimestampsCam[seq][ni+1]-tframe;
            else if(ni>0) // 当前为最后一帧，计算与上一帧的时间间隔
                T = tframe-vTimestampsCam[seq][ni-1];

            if(ttrack<T) // 跟踪耗时小于帧间间隔时，休眠补足时间间隔
                usleep((T-ttrack)*1e6); // 1e6
        }
        if(seq < num_seq - 1) // 处理下一序列（如果有）
        {
            cout << "Changing the dataset" << endl;

            SLAM.ChangeDataset();
        }
    }


    // Stop all threads
    SLAM.Shutdown(); // 结束

    // Tracking time statistics

    // Save camera trajectory
    // 记录当前系统时间（后续未被使用）
    std::chrono::system_clock::time_point scNow = std::chrono::system_clock::now();
    std::time_t now = std::chrono::system_clock::to_time_t(scNow);
    std::stringstream ss;
    ss << now;

    if (bFileName) // 是否有指定轨迹文件名 bFileName
    {
        const string kf_file =  "kf_" + string(argv[argc-1]) + ".txt";
        const string f_file =  "f_" + string(argv[argc-1]) + ".txt";
        SLAM.SaveTrajectoryEuRoC(f_file);
        SLAM.SaveKeyFrameTrajectoryEuRoC(kf_file);
    }
    else
    {
        SLAM.SaveTrajectoryEuRoC("CameraTrajectory.txt");
        SLAM.SaveKeyFrameTrajectoryEuRoC("KeyFrameTrajectory.txt");
    }

    sort(vTimesTrack.begin(),vTimesTrack.end()); // 对跟踪时间进行排序
    float totaltime = 0;
    for(int ni=0; ni<nImages[0]; ni++) // 跟踪时间累加
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages[0]/2] << endl; // 输出中位数跟踪耗时
    cout << "mean tracking time: " << totaltime/proccIm << endl; // 输出平均跟踪耗时

    return 0;
}

// 定义图像数据读取函数
void LoadImagesTUMVI(const string &strPathLeft, const string &strPathRight, const string &strPathTimes,
                vector<string> &vstrImageLeft, vector<string> &vstrImageRight, vector<double> &vTimeStamps)
{
    // 打印文件路径
    ifstream fTimes;
    cout << strPathLeft << endl;
    cout << strPathRight << endl;
    cout << strPathTimes << endl;
    // 打开图像时间戳文件
    fTimes.open(strPathTimes.c_str()); // 若 fTimes.eof() == true，则表示文件读取到末尾
    // 预留空间
    vTimeStamps.reserve(5000);
    vstrImageLeft.reserve(5000);
    vstrImageRight.reserve(5000);
    while(!fTimes.eof())
    {
        // 循环读取每一行
        string s;
        getline(fTimes,s);

        if(!s.empty()) // 非空行处理
        {
            if (s[0] == '#') // 跳过 ‘#’ 注释行
                continue;

            int pos = s.find(' ');              // 找出第一个空格的位置
            string item = s.substr(0, pos);     // 读取空格前的时间戳字符串

            // 拼接并存放左右图像完整路径
            vstrImageLeft.push_back(strPathLeft + "/" + item + ".png");
            vstrImageRight.push_back(strPathRight + "/" + item + ".png");

            double t = stod(item);          // string 转 double
            vTimeStamps.push_back(t/1e9);   // 存放单位从 ns 转为 s 的时间戳
        }
    }
}

// 定义 IMU 数据读取函数
void LoadIMU(const string &strImuPath, vector<double> &vTimeStamps, vector<cv::Point3f> &vAcc, vector<cv::Point3f> &vGyro)
{
    // 打开 IMU 时间戳文件
    ifstream fImu;
    fImu.open(strImuPath.c_str());
    // 预留空间
    vTimeStamps.reserve(5000);
    vAcc.reserve(5000);
    vGyro.reserve(5000);

    while(!fImu.eof()) // 若 fImu.eof() == true，则表示文件读取到末尾
    {
        // 循环读取每一行
        string s;
        getline(fImu,s);

        if (s[0] == '#') // 跳过 ‘#’ 注释行
            continue;

        if(!s.empty()) // 非空行处理
        {
            string item;
            size_t pos = 0;
            double data[7];
            int count = 0;
            // 读取以 ‘,’ 为分隔的数据（行末为空格）
            // 寻找第一个  ‘,’ 的位置，没有则返回末位
            while ((pos = s.find(',')) != string::npos) {
                item = s.substr(0, pos);     // 读取逗号前的字符串
                data[count++] = stod(item);  // string 转 double 并存放
                s.erase(0, pos + 1);         // 删除已读取部分和逗号
            }
            item = s.substr(0, pos);   // 读取剩余字符串
            data[6] = stod(item);      // string 转 double 并存放

            vTimeStamps.push_back(data[0]/1e9);                     // 存放单位从 ns 转为 s 的时间戳
            vAcc.push_back(cv::Point3f(data[4],data[5],data[6]));   // 存放加速度数据
            vGyro.push_back(cv::Point3f(data[1],data[2],data[3]));  // 存放角速度数据
        }
    }
}
