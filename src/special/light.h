#pragma once

#include <vector>

#include "../include/common_types.h"



class Light {
public:
    Light();
    Light(Config &config);

    enum LightFlag {
    LIGHT_NONE = 0,
    LIGHT_GREEN_FAR = 1,
    LIGHT_GREEN_NEAR = 2,
    LIGHT_RED_FAR = 3,
    LIGHT_RED_NEAR = 4,
    LIGHT_LOST = 5,
    };
    LightFlag flag_light;
    void check_light(const std::vector<DetectionTarget>& predict_result);
    void run_light(const void* src_img, const std::vector<DetectionTarget>& predict_result);

    
};
