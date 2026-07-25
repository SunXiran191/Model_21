#include "human.hpp"
#include <algorithm>
#include <cmath>
#include "../imgprocess/transform.hpp"

using namespace std;
using namespace cv;

Human::Human() : closest_offset_(0.0f), cx(0.0f), control_state_(ControlState::NONE)
{
    // 初始化状态机查表
    human_control_[0][0] = [this]()
    { control_state_ = ControlState::GO; }; // LEFT_FAR + LEFT: 直行
    human_control_[0][1] = [this]()
    { control_state_ = ControlState::SLOW; }; // LEFT_FAR + RIGHT: 减速
    human_control_[3][0] = [this]()
    { control_state_ = ControlState::SLOW; }; // RIGHT_FAR + LEFT: 减速
    human_control_[3][1] = [this]()
    { control_state_ = ControlState::GO; }; // RIGHT_FAR + RIGHT: 直行

    human_control_[1][0] = [this]()
    { control_state_ = ControlState::STOP; }; // LEFT_NEAR + LEFT: 停车
    human_control_[1][1] = [this]()
    { control_state_ = ControlState::STOP; }; // LEFT_NEAR + RIGHT: 停车
    human_control_[2][0] = [this]()
    { control_state_ = ControlState::STOP; }; // RIGHT_NEAR + LEFT: 停车
    human_control_[2][1] = [this]()
    { control_state_ = ControlState::STOP; }; // RIGHT_NEAR + RIGHT: 停车
}

Human::Human(Config &config) : closest_offset_(0.0f), cx(0.0f), control_state_(ControlState::NONE)
{
    this->human_near_threshold_ = config.human_near_threshold;
    this->human_control_y_threshold_ = config.human_control_y_threshold;
    this->human_speed_smooth_ = config.human_speed_smooth;
    this->human_plus_near_dist = config.human_plus_near_dist;

    // 初始化状态机查表
    human_control_[0][0] = [this]()
    { control_state_ = ControlState::GO; }; // LEFT_FAR + LEFT: 直行
    human_control_[0][1] = [this]()
    { control_state_ = ControlState::STOP; }; // LEFT_FAR + RIGHT: 减速
    human_control_[3][0] = [this]()
    { control_state_ = ControlState::STOP; }; // RIGHT_FAR + LEFT: 减速
    human_control_[3][1] = [this]()
    { control_state_ = ControlState::GO; }; // RIGHT_FAR + RIGHT: 直行

    human_control_[1][0] = [this]()
    { control_state_ = ControlState::STOP; }; // LEFT_NEAR + LEFT: 停车
    human_control_[1][1] = [this]()
    { control_state_ = ControlState::STOP; }; // LEFT_NEAR + RIGHT: 停车
    human_control_[2][0] = [this]()
    { control_state_ = ControlState::STOP; }; // RIGHT_NEAR + LEFT: 停车
    human_control_[2][1] = [this]()
    { control_state_ = ControlState::STOP; }; // RIGHT_NEAR + RIGHT: 停车
}

// ============================================================================
// 主入口：处理一帧行人数据
// ============================================================================
void Human::run_human(
    const vector<cv::Point> &human_point,
    const vector<cv::Point> &trajectory,
    int trajectory_size,
    float car_track_offset)
{
    // 无行人检测 → 直接返回
    if (human_point.empty())
    {
        control_state_ = ControlState::NONE;
        return;
    }

    cv::Point2f car_ipm = transf(IMAGE_W / 2.0f, IMAGE_H * 0.98f);
    cx = car_ipm.x;
    car_ipm_y = car_ipm.y;
    float ipm_dist = 0;

    // 默认无控制
    control_state_ = ControlState::NONE;

    human_dist_y = 0.0f;
    int max_y_ = 0, best_ped_x = 0;
    for (const auto &pt : human_point)
    {
        if (pt.y > max_y_)
        {
            max_y_ = pt.y;
            best_ped_x = pt.x;
        }
    }
    if (max_y_ > 0)
    {
        human_dist_y = static_cast<float>(IMAGE_H - max_y_);
    }

    // best_ped_x, max_y_ 已是 IPM 坐标（来自 t_human_point），无需再 transf
    ipm_dist = abs(max_y_ - car_ipm.y);

    if (ipm_dist < human_control_y_threshold_)
    {
        // 追踪
        track_human(human_point);

        if (trajectory_size < 3 || ped_tracks_.empty())
            return;

        // 只处理离底边最近,Y最大的行人
        Track *closest_track = nullptr;
        int max_y = -1;
        for (auto &track : ped_tracks_)
        {
            if (track.age >= 2 && track.prev_point.y > max_y)
            {
                max_y = track.prev_point.y;
                closest_track = &track;
            }
        }

        if (closest_track == nullptr)
            return;

        // 找轨迹中最接近该行人 Y 的点，获取对应赛道中线 X
        int closest_idx = 0;
        int min_dy = 100000;
        for (int i = 0; i < trajectory_size; ++i)
        {
            int dy = abs(trajectory[i].y - closest_track->prev_point.y);
            if (dy < min_dy)
            {
                min_dy = dy;
                closest_idx = i;
            }
        }
        // 将道路中线 X 平移到车实际行驶路径：叠加车相对中线起点的横向偏移
        int track_x = trajectory[closest_idx].x + static_cast<int>(car_track_offset);

        // 存储最近行人的原始数据
        closest_offset_ = static_cast<float>(closest_track->prev_point.x - track_x);

        Position pos = determinePosition(closest_track->prev_point.x, track_x,
                                         closest_track->prev_point.y);
        Direction dir = determineDirection(closest_track->speed.x);

        // 判断近→远过渡
        bool is_near = (pos == Position::LEFT_NEAR || pos == Position::RIGHT_NEAR);
        bool is_far = (pos == Position::LEFT_FAR || pos == Position::RIGHT_FAR);
        bool is_near_to_far = (closest_track->was_near && is_far);

        // 更新 was_near
        if (is_near)
        {
            closest_track->was_near = true;
        }
        else if (is_far)
        {
            closest_track->was_near = false;
        }

        // 近→远过渡：加速驶过
        if (is_near_to_far)
        {
            control_state_ = ControlState::SPEEDUP;
            return;
        }

        // 查表
        human_control_[static_cast<int>(pos)][static_cast<int>(dir)]();
        prev_direction_ = dir;
    }
}

// ============================================================================
// 行人追踪（最近邻匹配 + EMA 速度平滑）
// ============================================================================
void Human::track_human(const vector<cv::Point> &human_point)
{
    const float MATCH_MAX_DIST = 50.0f;

    for (auto &t : ped_tracks_)
        t.find = false;

    for (const auto &pt : human_point)
    {
        int best_idx = -1;
        float best_dist = MATCH_MAX_DIST;

        for (size_t i = 0; i < ped_tracks_.size(); ++i)
        {
            if (ped_tracks_[i].find)
                continue;
            float dx = pt.x - ped_tracks_[i].prev_point.x;
            float dy = pt.y - ped_tracks_[i].prev_point.y;
            float d = sqrt(dx * dx + dy * dy);
            if (d < best_dist)
            {
                best_dist = d;
                best_idx = static_cast<int>(i);
            }
        }

        if (best_idx >= 0)
        {
            auto &t = ped_tracks_[best_idx];
            cv::Point2f raw_v(pt.x - t.prev_point.x, pt.y - t.prev_point.y);
            float alpha = human_speed_smooth_;
            t.speed = cv::Point2f(
                alpha * t.speed.x + (1.f - alpha) * raw_v.x,
                alpha * t.speed.y + (1.f - alpha) * raw_v.y);
            t.prev_point = pt;
            t.age++;
            t.miss_count = 0;
            t.find = true;
        }
        else
        {
            Track new_track;
            new_track.prev_point = pt;
            new_track.speed = cv::Point2f(0, 0);
            new_track.age = 1;
            new_track.miss_count = 0;
            new_track.find = true;
            new_track.was_near = false;
            ped_tracks_.push_back(new_track);
        }
    }

    // 清理过期 track
    static const int MAX_MISS_COUNT = 12;
    static const int MAX_AGE = 120;
    ped_tracks_.erase(
        remove_if(ped_tracks_.begin(), ped_tracks_.end(),
                  [](const Track &t)
                  {
                      return (!t.find && t.age < 3) ||
                             (t.miss_count > MAX_MISS_COUNT) ||
                             (t.age > MAX_AGE);
                  }),
        ped_tracks_.end());

    // 对未匹配的 track 进行速度预测外推
    for (auto &t : ped_tracks_)
    {
        if (!t.find)
        {
            t.miss_count++;
            if (t.age >= 3)
            {
                t.prev_point.x += static_cast<int>(t.speed.x);
                t.prev_point.y += static_cast<int>(t.speed.y);
            }
        }
    }
}

// ============================================================================
// 状态判定
// ============================================================================
Human::Position Human::determinePosition(int ped_x, int track_x, int ped_y) const
{
    // 基准 = 0.7×赛道中线 + 0.3×车位置
    float x_ref = 0.7f * track_x + 0.3f * cx;
    int offset = ped_x - static_cast<int>(x_ref); // 正=行人在基准右侧, 负=左侧

    // 线性阈值：行人越靠近车NEAR阈值越大
    int dist_y = abs(IMAGE_H - ped_y);
    float scale = 1 - clipf(dist_y / human_control_y_threshold_, 0.1f, 1.0f);
    float dyn_threshold = human_near_threshold_ + human_plus_near_dist * scale;

    if (offset < -dyn_threshold)
    {
        return Position::LEFT_FAR;
    }
    else if (offset > dyn_threshold)
    {
        return Position::RIGHT_FAR;
    }
    else if (offset < 0)
    {
        return Position::LEFT_NEAR;
    }
    else
    {
        return Position::RIGHT_NEAR;
    }
}

Human::Direction Human::determineDirection(float speed_x) const
{
    const float DIR_DEADZONE = 0.1f;

    if (speed_x > DIR_DEADZONE)
        return Direction::RIGHT;
    if (speed_x < -DIR_DEADZONE)
        return Direction::LEFT;

    // 死区内：沿用上一帧方向，防止检测噪声导致左右跳变
    return prev_direction_;
}
