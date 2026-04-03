#include "human.h"

Human::Human(const Config* config) : flag_human(HumanFlag::HUMAN_NONE) {
    (void)config;
}

void Human::check_human(const std::vector<DetectionTarget>& predict_result) {
    (void)predict_result;
}

void Human::run_human(const void* src_img, const std::vector<DetectionTarget>& predict_result) {
    (void)src_img;
    (void)predict_result;
}
