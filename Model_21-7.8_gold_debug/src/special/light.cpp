#include "light.hpp"
#include <cmath>
#include "../imgprocess/transform.hpp"
using namespace std;
using namespace cv;

// ── 辅助函数：灯色 → 带 emoji 的字符串 ──
static const char *colorToString(Light::LightColor c)
{
    switch (c)
    {
    case Light::COLOR_RED:
        return "🔴 RED";
    case Light::COLOR_GREEN:
        return "🟢 GREEN";
    case Light::COLOR_YELLOW:
        return "🟡 YELLOW";
    default:
        return "UNKNOWN";
    }
}

Light::Light()
{
}

Light::Light(Config &config)
{
    confidence_threshold = config.confidence_threshold;
    stop_distance_m = config.light_stop_distance_m;
}

void Light::check_light_color(const std::vector<PredictResult> &predict_result)
{
    bool has_green = false, has_yellow = false, has_red = false;
    for (const auto &obj : predict_result)
    {
        if (obj.score < confidence_threshold)
            continue;
        switch (obj.class_id)
        {
        case CLASS_ID_LIGHT_GREEN:
            has_green = true;
            break;
        case CLASS_ID_LIGHT_YELLOW:
            has_yellow = true;
            break;
        case CLASS_ID_LIGHT_RED:
            has_red = true;
            break;
        default:
            break;
        }
    }

    if (has_red || has_yellow || has_green)
    {
        lost_cnt = 0;
        // 优先级：红 > 黄 > 绿
        if (has_red)
            light_color = COLOR_RED;
        else if (has_yellow)
            light_color = COLOR_YELLOW;
        else if (has_green)
            light_color = COLOR_GREEN;
    }
    else
    {
        if (light_color != COLOR_NONE)
        {
            ++lost_cnt;
            if (lost_cnt >= LOST_FRAMES_THRESH)
            {
                light_color = COLOR_NONE;
                lost_cnt = 0;
            }
        }
    }
    if (light_color != COLOR_NONE)
    {
        printf("%s\n", colorToString(light_color));
    }
}

void Light::run_light(const std::vector<PredictResult> &predict_result,
                      const std::vector<cv::Point> &t_light_point,
                      const std::vector<cv::Point> &t_zebra_point,
                      const int &fps,
                      const float &car_speed_mps,
                      float &ctrl_speed)
{
    ++frame_count;

    cv::Point2f car_base_ipm = transf(IMAGE_W / 2.0f, IMAGE_H * 0.95f);
    float cx = car_base_ipm.x;
    float cy = car_base_ipm.y;

    // 使用斑马线（停止线）坐标计算距离，而非灯光坐标
    // 斑马线底边中点 y 坐标代表停止线位置
    float dist = 999.0f;
    if (!t_zebra_point.empty())
    {
        if ((t_zebra_point[0].x < IMAGE_W - 1 && t_zebra_point[0].x > 0) && (t_zebra_point[0].y < IMAGE_H - 1 && t_zebra_point[0].y > 0))
        {
            dist = (cy - t_zebra_point[0].y) / PIXPERMETER;
            printf("dist:%.0f \n", dist);
        }
    }

    if (light_color == COLOR_NONE)
    {
        light_action = ACTION_GO;
        yellow_start_frame = -1;
        remaining_yellow_sec = 0.0f;
    }
    else
    {
        switch (light_color)
        {
        case COLOR_GREEN:
            yellow_start_frame = -1;
            remaining_yellow_sec = 0.0f;
            light_action = ACTION_GO;
            break;

        case COLOR_RED:
            if (dist <= stop_distance_m)
            {
                yellow_start_frame = -1;
                remaining_yellow_sec = 0.0f;
                light_action = ACTION_STOP;
            }
            break;

        case COLOR_YELLOW:
            if (dist <= stop_distance_m)
            {
                yellow_start_frame = -1;
                remaining_yellow_sec = 0.0f;
                light_action = ACTION_STOP;
            }
            break;

        default:
            light_action = ACTION_GO;
            break;
        }
    }

    switch (light_action)
    {
    case ACTION_GO:
        break;

    case ACTION_CAUTION:
        ctrl_speed = (dist <= stop_distance_m) ? 0.0f : ctrl_speed * slow_speed_ratio;
        break;

    case ACTION_STOP:
        ctrl_speed = 0.0f;
        break;
    }
}