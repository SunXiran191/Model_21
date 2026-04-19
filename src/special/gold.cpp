#include "gold.h"

using namespace std;
using namespace cv;

Gold::Gold() {}
Gold::Gold(Config &config) {}

void Gold::check_gold(const std::vector<PredictResult> &predict_result)
{
    bool is_gold_detected = false;
    static int find_cnt = 0;
    static int lost_cnt = 0;

    for (const auto &target : predict_result)
    {
        if (target.class_id == CLASS_ID_GOLD && target.score > PREDICT_THRESH)
        {
            is_gold_detected = true;
            break;
        }
    }

    // 状态转换逻辑
    if (gold_flag == GoldFlag::GOLD_NONE)
    {
        if (is_gold_detected)
        {
            find_cnt++;
            if (find_cnt >= PREDICT_THRESH)
            {
                gold_flag = GoldFlag::GOLD_DETECTED;
                logger.info("GOLD detected!"); // 确保logger已正确初始化//logger部分补充
                find_cnt = 0;
            }
        }
    }
    else if (gold_flag == GoldFlag::GOLD_DETECTED)
    {
        if (!is_gold_detected)
        {
            gold_flag = GoldFlag::GOLD_LOST;
            lost_cnt++;
        }
    }
    else if (gold_flag == GoldFlag::GOLD_LOST)
    {
        if (!is_gold_detected && lost_cnt > LOST_THRESH)
        {
            gold_flag = GoldFlag::GOLD_NONE;
            lost_cnt = 0;
        }
    }
}

void Gold::run_gold(const void *src_img,
                    const std::vector<PredictResult> &predict_result,
                    const std::vector<POINT> &CV_pointsOrigin,
                    std::vector<POINT> &trackPoints,
                    int CV_pointsOrigin_size,
                    int trackPoints_size)
{
    found_valid_gold = false;
    static int cnt_getgold = 0;
    static int cnt_lostgold = 0;

    if (gold_flag == GoldFlag::GOLD_DETECTED)
    {
        // 设定取点距离
        int dist_forward = 40;
        int dist_backward = 40;
        int start_idx = -1;
        int end_idx = -1;

        // 金币前后赛道点截断
        for (int i = 0; i < CV_pointsOrigin_size; ++i)
        {
            if (start_idx == -1 && CV_pointsOrigin[i].y <= general.clipf(t_gold_point.y + dist_forward, 0, src_img.rows - 1))
            {
                start_idx = i;
            }

            if (end_idx == -1 && CV_pointsOrigin[i].y <= general.clipf(t_gold_point.y - dist_backward, 0, src_img.rows - 1))
            {
                end_idx = i;
                break;
            }
        }

        if (start_idx != -1 && end_idx != -1 && start_idx < end_idx)
        {
            gold_flag = GoldFlag::GOLD_LOCATED;
        }
    }
    else if (gold_flag == GoldFlag::GOLD_LOCATED)
    {
        POINT p_start = CV_pointsOrigin[start_idx];
        POINT p_end = CV_pointsOrigin[end_idx];

        // POINT p_control;
        // p_control.x = 2 * p_gold.x - (p_start.x + p_end.x) / 2;
        // p_control.y = 2 * p_gold.y - (p_start.y + p_end.y) / 2;

        std::vector<POINT> bezier_segment = TrajectoryFitter::FitTrajectory_Bezier_2d(p_start, p_gold, p_end, 20);

        // 将拟合曲线插入原赛道点集
        trackPoints.erase(trackPoints.begin() + start_idx, trackPoints.begin() + end_idx);
        trackPoints.insert(trackPoints.begin() + start_idx, bezier_segment.begin(), bezier_segment.end());

        logger.info("Gold detected! Trajectory replanned.");

        if (best_gold.y1 >= trackPoints.front().y)
        {
            cnt_getgold++;
        }
        if (cnt_getgold >= 10)
        {
            gold_flag = GoldFlag::GOLD_GET;
            cnt_getgold = 0;
            logger.info("Gold got!");
        }
    }
    else if (gold_flag == GoldFlag::GOLD_GET)
    {
        for (const auto &target : predict_result)
        {
            if (!CLASS_ID_GOLD) // 这是改过后的是否检测到金币的布尔变量
            {
                cnt_get_lostgold++;
            }
        }
        if (cnt_lostgold >= 10)
        {
            gold_flag = GoldFlag::GOLD_NONE;
            cnt_lostgold = 0;
            logger.info("Gold none!");
        }
    }
}