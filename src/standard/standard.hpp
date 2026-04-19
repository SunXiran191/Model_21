#pragma once

#include <vector>
#include <cmath>
#include <iostream>
#include <opencv2/core.hpp>

#include "../../res/include/common.hpp"
#include "general.hpp"
#include "../../res/include/aiget.hpp"
#include "../common/utils.hpp"
#include "../special/car.hpp"
#include "../special/gold.hpp"
#include "../special/human.hpp"
#include "../special/light.hpp"
#include "../imgprocess/LineTracker.hpp"

// 实时数据，任务逻辑输出数据
struct TaskData
{
    cv::Mat img;
    std::chrono::time_point<std::chrono::_v2::steady_clock, std::chrono::duration<long int, std::ratio<1, 1000000000>>> timestamp;
    bool beep_enable = 0;
    float speed = 0.0f;
    float angle = 0.0f;
    bool bend_flag = false; //弯道标志
};

class Scene_status
{
public:
    bool GoldScene;
    bool CarScene;
    bool HumanScene;
    bool LightScene;

    Scene_status();
    bool all();
    // bool none();
};

class Standard
{
public:
    Standard();
    Standard(Config config);

    enum TrackState
    {
        TRACK_MIDDLE = 0,
        TRACK_AI_MIDDLE = 1
    };
    TrackState trackstate = TRACK_MIDDLE;

    LineTracker tracker;
    Gold gold_;
    Car car_;
    Human human_;
    Light light_;

    enum Scene_status
    {
        Normal_status = 0,
        Gold_status = 1,
        Car_status = 2,
        Human_status = 3,
        Light_status = 4
    };
    Scene_status scene_status = Normal_status;

    // 定义 class_id映射关系
    constexpr int CLASS_ID_GOLD = 0;
    constexpr int CLASS_ID_CAR = 1;
    constexpr int CLASS_ID_HUMAN = 2;
    constexpr int CLASS_ID_LIGHT = 3;

    vector<aiget::PredictResult> gold_results_;
    vector<aiget::PredictResult> car_results_;
    vector<aiget::PredictResult> human_results_;
    vector<aiget::PredictResult> light_results_;

    vector<POINT> gold_point;
    vector<POINT> car_point;
    vector<POINT> human_point;
    vector<POINT> light_point;

    vector<POINT> t_gold_point;
    vector<POINT> t_car_point;
    vector<POINT> t_human_point;
    vector<POINT> t_light_point;

    vector<POINT> trackPoints_CV;     // 传统cv赛道点集
    vector<POINT> filtered_line_CV;   // 传统cv赛道拟合后点集
    vector<POINT> t_trackPoints_CV;   // 透视变换后传统cv赛道点集
    vector<POINT> s_t_trackPoints_CV; // 等距采样后传统cv赛道点集
    vector<POINT> t_angle_CV;         // 计算角度后点集

    vector<POINT> s_t_trackPoints_APF; // 人工势场法赛道点集

    vector<POINT> trackPoints_AI;     // AI分割赛道点集
    vector<POINT> t_trackPoints_AI;   // 透视变换后AI分割赛道点集
    vector<POINT> s_t_trackPoints_AI; // 等距采样后AI分割赛道点集
    vector<POINT> t_angle_AI;         // 计算角度后点集
    vector<POINT> t_CenterEdge;       // 计算曲率点集

    int trackPoints_CV_size;     // 传统cv赛道点集
    int filtered_line_CV_size;   // 传统cv赛道拟合后点集
    int t_trackPoints_CV_size;   // 透视变换后传统cv赛道点集
    int s_t_trackPoints_CV_size; // 等距采样后传统cv赛道点集
    int t_angle_CV_size;         // 计算角度后点集

    int s_t_trackPoints_APF_size; // 人工势场法赛道点集

    int trackPoints_AI_size;     // AI分割赛道点集
    int t_trackPoints_AI_size;   // 透视变换后AI分割赛道点集
    int s_t_trackPoints_AI_size; // 等距采样后AI分割赛道点集
    int t_angle_AI_size;         // 计算角度后点集
    int t_CenterEdge_size;       // 计算曲率点集大小

    TaskData run(const cv::Mat &src_img, const std::vector<PredictResult> &predict_result,
                 double pitch_angle);

private:
    float cx = COLSIMAGE / 2.0f; // 车轮对应点 (纯跟踪起始点)
    float cy = ROWSIMAGE * 0.999f;
    int aim_index_far;
    int aim_index_near;
    // double pixel_per_meter = 222.222; // 一米在画面中所占的像素数
    // double SAMPLE_DIST = 0.02;        // 采样间距 单位m
    // double ROAD_WIDTH = 0.45;         // 道路真实宽度 单位m
    // double dist = pixel_per_meter * ROAD_WIDTH / 2.0f;
    // double wheelbase_val = 0.2;    // 车轴距
    double car_length = 0.316; // 车身长

    float aim_distance_f;
    float aim_distance_n;
    float aim_angle_p_k;
    float aim_angle_p;
    float aim_angle_d;

public:
    float aim_speed = 0.f;
    float aim_angle = 0.0f;      // 偏差量
    float aim_angle_last = 0.0f; // 偏差量 上一帧
    float aim_sigma = 0.0f;      // 偏差方差、

    Logger logger = Logger("Standard");
};
