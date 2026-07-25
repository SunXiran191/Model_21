#pragma once

#include <vector>
#include <functional>
#include <opencv2/core.hpp>
#include "../../res/configs/param.hpp"
#include "../common/utils.hpp"

class Human
{
public:
    // 方向状态
    enum class Direction
    {
        LEFT = 0,
        RIGHT = 1
    };

    // 位置状态
    enum class Position
    {
        LEFT_FAR = 0,
        LEFT_NEAR = 1,
        RIGHT_NEAR = 2,
        RIGHT_FAR = 3
    };

    // 控制状态
    enum class ControlState
    {
        NONE = 0,    // 无行人 → 正常行驶
        GO = 1,      // 行人在远区
        SPEEDUP = 2, // 行人从近区到远区：加速驶过
        SLOW = 3,    // 左远向右，右远向左 → 减速
        STOP = 4,    // 行人近区横穿 → 停车
    };

    bool human_stopted = false;
    int human_dist_y = 0; // 行人距离赛道底部的 Y 坐标

    Human();
    Human(Config &config);

    // 主入口：处理一帧行人检测
    // car_track_offset: 车实际位置相对中线起点的横向偏移，用于将判断基准从道路中线平移到车实际路径
    void run_human(const std::vector<cv::Point> &human_point, const std::vector<cv::Point> &trajectory,
                   int trajectory_size,
                   float car_track_offset = 0.0f);

    // 获取当前帧控制状态
    ControlState getControlState() const { return control_state_; }

    // 重置控制状态（轨迹无效时调用，防止残留上一帧的非NONE值）
    void resetControlState() { control_state_ = ControlState::NONE; }

    // 获取最近行人的原始数据（供模糊控制器使用）
    float getClosestOffset() const { return closest_offset_; } // ped_x - track_x
    float getClosestOffsetRatio() const
    {
        float abs_off = (closest_offset_ >= 0.0f) ? closest_offset_ : -closest_offset_;
        return abs_off / static_cast<float>(human_near_threshold_);
    }

private:
    // 行人追踪
    struct Track
    {
        cv::Point prev_point; // 上一帧位置
        cv::Point2f speed;    // EMA平滑后的速度
        int age;              // 连续追踪帧数
        int miss_count;       // 连续未匹配帧数
        bool find;            // 当前帧是否匹配到
        bool was_near;        // 上一帧是否在近区
    };

    std::vector<Track> ped_tracks_;

    // 状态机查表
    std::function<void()> human_control_[4][2]; // human_control_[pos][dir]

    int human_near_threshold_;      // 近/远区分界阈值
    int human_control_y_threshold_; // 行人停车 Y 坐标阈值
    float human_speed_smooth_;      // 行人速度 EMA 平滑系数
    int human_plus_near_dist;

    // 当前帧行人
    float closest_offset_; // ped_x - track_x
    float cx;              // 车体 IPM X 坐标（用于 FAR/NEAR 判断基准）
    float car_ipm_y;       // 车体 IPM Y 坐标（用于动态阈值缩放）

    // 输出状态
    ControlState control_state_;
    Direction prev_direction_ = Direction::LEFT; // 上一帧方向（防死区跳变）

    // 行人跟踪
    void track_human(const std::vector<cv::Point> &human_point);

    // 状态判定
    Position determinePosition(int ped_x, int track_x, int ped_y) const;
    Direction determineDirection(float speed_x) const;
};
