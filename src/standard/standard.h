#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include "../include/common_types.h"
#include "../special/car.h"
#include "../special/gold.h"
#include "../special/human.h"
#include "../special/light.h"

// 实时数据，任务逻辑输出数据
struct TaskData
{
    cv::Mat img;
    std::chrono::time_point<std::chrono::_v2::steady_clock, std::chrono::duration<long int, std::ratio<1, 1000000000>>> timestamp;
    bool beep_enable = 0;
    float speed = 0.0f;
    float angle = 0.0f;
};

class Standard
{

public:
    enum TrackState
    {
        TRACK_MIDDLE = 0;
        TRACK_AI_MIDDLE = 1;
    };
    TrackState trackstate = TrackState::TRACK_MIDDLE;

    enum Scene
    {
        Normal = 0;
        Gold = 1;
        Car = 2;
        Human = 3;
        Light = 4;
    };
    Scene scene = Scene::Normal;

public:
    Standard();
    Standard(Config config);

    vector<POINT> CV_pointsOrigin;   // 赛道传统巡线点集
    vector<POINT> AI_pointsOrigin;   // 赛道AI巡线点集
    vector<POINT> pointsTaskout;     // 任务拟合后点集
    vector<POINT> t_CV_pointsOrigin; // 逆透视赛道传统巡线点集
    vector<POINT> t_AI_pointsOrigin; // 逆透视赛道AI巡线点集
    vector<POINT> t_pointsTaskout;   // 逆透视任务拟合后点集
    vector<POINT> trackPoints;      // 逆透视赛道中心线点集

    int CV_pointsOrigin_size;
    int AI_pointsOrigin_size;
    int pointsTaskout_size;
    int t_CV_pointsOrigin_size;
    int t_AI_pointsOrigin_size;
    int t_pointsTaskout_size;
    int trackPoints_size;

    vector<int> _elem_order;
    int order_index = 0;
    vector<float> _aim_dis_n_order;
    vector<float> _aim_dis_f_order;
    vector<int> _speed_order;
    vector<float> _aim_angle_p_order;
    vector<float> _aim_angle_d_order;
    vector<int> obstacel_order_index;

    TaskData
    run(const cv::Mat &src_img, const std::vector<DetectionTarget> &predict_result,
        double pitch_angle);

private:
    double DynamicAimDisCal();
};
