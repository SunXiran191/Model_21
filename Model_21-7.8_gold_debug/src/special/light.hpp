#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include "../../res/configs/param.hpp"
#include "../../res/include/aiget.hpp"
#include "../common/utils.hpp"
#include "../imgprocess/transform.hpp"

class Light
{
public:
    Light();
    Light(Config &config);

    enum LightColor
    {
        COLOR_NONE = 0,
        COLOR_GREEN,
        COLOR_YELLOW,
        COLOR_RED,
    };

    enum Action
    {
        ACTION_GO = 0,  // 正常通行 / 加速通过
        ACTION_CAUTION, // 减速
        ACTION_STOP,    // 停车
    };

    // ── 对外可读状态 ──
    LightColor light_color = COLOR_NONE; // 当前帧灯色
    Action light_action = ACTION_GO;     // 当前决策
    float dist_to_zebra_m;               // 车→斑马线距离(米)
    float remaining_yellow_sec = 0.0f;   // 黄灯剩余秒数

    float confidence_threshold = 0.44f; // 检测置信度阈值
    int fps = 30;                       // 默认帧率

    /// 从检测结果中识别当前灯色（每帧调用）
    void check_light_color(const std::vector<PredictResult> &predict_result);

    /// 根据灯色执行决策（灯色有效时调用）
    void run_light(const std::vector<PredictResult> &predict_result,
                   const std::vector<cv::Point> &t_light_point,
                   const std::vector<cv::Point> &t_zebra_point,
                   const int &fps,
                   const float &car_speed_mps,
                   float &ctrl_speed);

    float stop_distance_m;    // 停车时距斑马线距离

    float caution_distance_m = 0.25f; // 开始减速距离
    float slow_speed_ratio = 0.50f;   // 减速阶段目标速度比例
    float boost_speed_ratio = 1.20f;  // 黄灯加速通过速度比例
    int lost_cnt = 0;                 // 连续丢失帧数
    int LOST_FRAMES_THRESH = 5;       // 连续丢失多少帧后清空灯色

private:
    int yellow_start_frame = -1;
    int frame_count = 0;
};
