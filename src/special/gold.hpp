#pragma once

#include <vector>
#include "../TrajectoryFitter.hpp"
#include "../standard/general.hpp"

class Gold
{
public:
    Gold();
    Gold(Config &config);

    // struct GoldDetection
    // {
    //     int x1; // 左上角
    //     int y1;
    //     int x2; // 右下角
    //     int y2;
    //     float score; // 置信度
    //     int class_id;
    // };
    // GoldDetection gold_detection;

    enum GoldFlag
    {
        GOLD_NONE = 0,
        GOLD_DETECTED = 1,
        GOLD_LOCATED = 2,
        GOLD_GET = 3,
    };
    GoldFlag gold_flag = GoldFlag::GOLD_NONE;

    void check_gold(const std::vector<PredictResult> &predict_result);
    void run_gold(const void *src_img, const std::vector<PredictResult> &predict_result, vector<POINT> CV_pointsOrigin, vector<POINT> trackPoints, int CV_pointsOrigin_size, int trackPoints_size);

private:
    bool found_valid_gold;
    PredictResult best_gold;
};
