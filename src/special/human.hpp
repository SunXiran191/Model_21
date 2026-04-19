#pragma once

#include <vector>

#include "../include/common_types.h"



class Human {
public:
    Human();
    Human(Config &config);

    enum  HumanFlag 
    {
        HUMAN_NONE = 0,
        HUMAN_DETECT = 1,
        HUMAN_LOST = 2,
    };
    HumanFlag human_flag = HumanFlag::HUMAN_NONE;

    void check_human(const std::vector<PredictResult>& predict_result);
    void run_human(const void* src_img, const std::vector<PredictResult>& predict_result);

private:
};
