#include "objects.h"

using namespace std;
using namespace cv;

Objects::Objects() {}
Objects::Objects(Config &config) {}

void Objects::check_objects(const std::vector<PredictResult> &predict_result)
{
    bool is_human_detected = false;
    static int find_cnt = 0;
    static int lost_cnt = 0;

    for (const auto &target : predict_result)
    {
        if (target.class_id == CLASS_ID_HUMAN && target.score > PREDICT_THRESH)
        {
            is_human_detected = true;
            break;
        }
    }

    if (human_flag == HumanFlag::HUMAN_NONE)
    {
        if (is_human_detected)
        {
            find_cnt++;
            if (find_cnt >= PREDICT_THRESH)
            {
                human_flag = HumanFlag::HUMAN_DETECT;
                logger.info("HUMAN detected!");
                find_cnt = 0;
            }
        }
    }
    else if (human_flag == HumanFlag::HUMAN_DETECT)
    {
        if (!is_human_detected)
            human_flag = HumanFlag::HUMAN_LOST;
    }
    else if (human_flag == HumanFlag::HUMAN_LOST)
    {
        if (!is_human_detected)
            human_flag = HumanFlag::HUMAN_NONE;
    }
}

void Human::run_human(const cv::Mat &src_img,
                      const std::vector<PredictResult> &predict_result,
                      const std::vector<POINT> &CV_pointsOrigin,
                      std::vector<POINT> &trackPoints,
                      int CV_pointsOrigin_size,
                      int trackPoints_size)
{
    if (human_flag != HumanFlag::HUMAN_DETECT)
        return;

    bool found = false;
    PredictResult best_human;
    for (const auto &target : predict_result)
    {
        if (target.class_id == CLASS_ID_HUMAN && target.score > 0.6f)
        {
            if (!found || target.y2 > best_human.y2)
            {
                best_human = target;
                found = true;
            }
        }
    }
    if (!found)
        return;

    POINT p_human = {(best_human.x1 + best_human.x2) / 2,
                     (best_human.y1 + best_human.y2) / 2};

    int dist_forward = 50;
    int dist_backward = 50;
    int start_idx = -1;
    int end_idx = -1;

    for (int i = 0; i < CV_pointsOrigin_size; ++i)
    {
        if (start_idx == -1 && CV_pointsOrigin[i].y <= p_human.y + dist_forward)
            start_idx = i;
        if (end_idx == -1 && CV_pointsOrigin[i].y <= p_human.y - dist_backward)
        {
            end_idx = i;
            break;
        }
    }

    if (start_idx < 0 || end_idx < 0 || start_idx >= end_idx)
        return;

    int img_center_x = src_img.cols / 2;
    int avoid_dir = (p_human.x < img_center_x) ? 1 : -1;
    int lateral_offset = 70;
    POINT p_avoid = {p_human.x + avoid_dir * lateral_offset, p_human.y};

    POINT p_start = CV_pointsOrigin[start_idx];
    POINT p_end = CV_pointsOrigin[end_idx];
    std::vector<POINT> bezier_segment =
        TrajectoryFitter::FitTrajectory_Bezier_2d(p_start, p_avoid, p_end, 24);

    trackPoints.erase(trackPoints.begin() + start_idx, trackPoints.begin() + end_idx);
    trackPoints.insert(trackPoints.begin() + start_idx, bezier_segment.begin(), bezier_segment.end());

    // Keep HUMAN_DETECT until human disappears
}