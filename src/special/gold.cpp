#include "gold.h"

Gold::Gold(const Config* config) : flag_gold(GoldFlag::GOLD_NONE) {
    (void)config;
}

void Gold::check_gold(const std::vector<DetectionTarget>& predict_result) {
    (void)predict_result;
}

void Gold::run_gold(const void* src_img, const std::vector<DetectionTarget>& predict_result) {
    (void)src_img;
    (void)predict_result;
}
