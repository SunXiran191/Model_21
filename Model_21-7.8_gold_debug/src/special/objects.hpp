#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include "../../res/configs/param.hpp"
#include "../../res/include/aiget.hpp"
#include "../common/utils.hpp"

// 障碍物避障目标点信息
struct ObjectsTargetInfo
{
    cv::Point pt; // 目标点位置
    float y_low;  // 屏蔽下y坐标
    float y_high; // 屏蔽上y坐标
};

// 金币夹角信息（角度 + 对应的Y范围）
struct GoldAngleInfo
{
    float angle;  // 内角度数（180°=直道，偏离越大弯越急）
    float y_low;  // 该角对应三连点的Y下界（最远）
    float y_high; // 该角对应三连点的Y上界（最近）
};

class Objects
{
public:
    Objects();
    Objects(Config &config);

    std::vector<ObjectsTargetInfo> targets;

    // 主入口：处理金币 + 车辆避障
    void processAllObjects(
        cv::Mat &src_img,
        const std::vector<cv::Point> &t_gold_point,
        const std::vector<cv::Point> &t_car_point,
        std::vector<cv::Point> &trajectory,
        int &trajectory_size);

    // ====== 金币夹角（转弯标志） ======
    std::vector<GoldAngleInfo> gold_angles; // 所有金币序列内角（含Y范围）
    bool gold_angle_flag = false;           // 金币夹角有效标志
    float gold_seq_y_min = 0.0f;            // 序列Y范围下界
    float gold_seq_y_max = 0.0f;            // 序列Y范围上界

    // ====== 配置参数 ======
    int gold_dist_forward;
    int gold_dist_backward;
    int car_dist_forward;
    int car_dist_backward;
    int car_avoid_dist;
    int max_gold_track_dist;
    float confidence_threshold;

    // ====== 静态工具方法 ======
    static std::vector<cv::Point> generateRubberBandCurve(
        const std::vector<cv::Point> &input_points,
        int window_size = 15, int iterations = 5);

    static std::vector<cv::Point> resampleEquidistant(
        const std::vector<cv::Point> &input, float step_dist);

private:
    // 车辆避障点计算
    cv::Point computeCarAvoidPoint(
        const cv::Point &car_center,
        const std::vector<cv::Point> &trajectory,
        int trajectory_size);

    // 金币夹角序列计算
    void computeGoldAngleSeq(
        const std::vector<ObjectsTargetInfo> &gold_targets);

    // ====== 车辆避障方向迟滞（防左右跳变） ======
    int avoid_dir_last_ = 0;
    int avoid_dir_lock_cnt_ = 0;
    int avoid_dir_dead_zone = 0;
    int avoid_dir_flip_thresh = 2;
    float avoid_x_ema_ = -1.0f;
    float avoid_ema_alpha_ = 0.3f;
};