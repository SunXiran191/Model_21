#pragma once

#include <vector>

#include "../include/common_types.h"



class Human {
public:
    Human();
    Human(Config &config);

    enum  HumanFlag {
    HUMAN_NONE = 0,
    HUMAN_DETECT = 1,
    HUMAN_LOST = 2,
    };
    void check_human(const std::vector<DetectionTarget>& predict_result);
    void run_human(const void* src_img, const std::vector<DetectionTarget>& predict_result);

    HumanFlag flag_human;
};
