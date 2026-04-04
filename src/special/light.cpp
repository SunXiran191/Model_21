#include "human.h"

using namespace std;
using namespace cv;

Light::Light(){}
Light::Light(Config &config){

}

void Light::check_light(const std::vector<PredictResult>& predict_result) {
    bool is_light_detected = false;

    for (const auto& target : predict_result) {
        if (target.class_id == light) {
            is_light_detected = true;
            break;
        }
    }

    if (is_light_detected) {
        flag_light = LightFlag::LIGHT_DETECT;
    } else {
        if (flag_light == LightFlag::LIGHT_DETECT) {
            flag_light = LightFlag::LIGHT_LOST;
        } else if (flag_light == LightFlag::LIGHT_LOST) {
            flag_light = LightFlag::LIGHT_NONE;
        }
    }

}