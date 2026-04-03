#pragma once

#include <vector>



class Gold {
public:
    Gold();
    Gold(Config &config);


    enum  GoldFlag {
    GOLD_NONE = 0,
    GOLD_GET = 1,
    GOLD_LOST = 2,
};
    void check_gold(const std::vector<DetectionTarget>& predict_result);
    void run_gold(const void* src_img, const std::vector<DetectionTarget>& predict_result);

    GoldFlag flag_gold;
};
