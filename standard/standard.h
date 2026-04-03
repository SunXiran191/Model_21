#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include "../include/common_types.h"
#include "../special/car.h"
#include "../special/gold.h"
#include "../special/human.h"
#include "../special/light.h"

//实时数据，任务逻辑输出数据
struct TaskData {
    cv::Mat img;
    std::chrono::time_point<std::chrono::_v2::steady_clock,std::chrono::duration<long int,std::ratio<1,1000000000>>> timestamp;
    bool beep_enable = 0;
    float speed = 0.0f;
    float angle = 0.0f;
};

class Standard {
public:
    Standard();
    Standard(Config config);

    enum TrackState {
    TRACK_MIDDLE = 0;
    TRACK_AI_MIDDLE = 1;
    };
    TrackState trackstate = TrackState::TRACK_MIDDLE;
    enum Scene {
    Normal = 0;
    Gold = 1;
    Car = 2;
    Human = 3;
    Light = 4;
    };
    Scene scene = Scene::Normal;

    TaskData run(const cv::Mat& src_img, const std::vector<DetectionTarget>& predict_result,
                 double pitch_angle);


private:


};
