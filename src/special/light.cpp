#include "light.h"

Light::Light(const Config* config) : flag_light(LightFlag::LIGHT_NONE) {
    (void)config;
}

void Light::check_light(const std::vector<DetectionTarget>& predict_result) {
    (void)predict_result;
}

void Light::run_light(const void* src_img, const std::vector<DetectionTarget>& predict_result) {
    (void)src_img;
    (void)predict_result;
}
