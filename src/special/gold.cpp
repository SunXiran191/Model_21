#include "gold.h"

using namespace std;
using namespace cv;


Gold::Gold(){}
Gold::Gold(Config &config){

}
void Gold::check_gold(const std::vector<PredictResult>& predict_result) {
    bool is_gold_toget = false;

    for (const auto& target : predict_result) {
        if (target.class_id == gold) {
            is_gold_toget = true;
            break;
        }
    }

    if (is_gold_toget) {
        flag_gold = GoldFlag::GOLD_GET;
    } else {
        if (flag_gold == GoldFlag::GOLD_GET) {
            flag_gold = GoldFlag::GOLD_LOST;
        } else if (flag_gold == GoldFlag::GOLD_LOST) {
            flag_gold = GoldFlag::GOLD_NONE;
        }
    }
}

void Gold::run_gold(const void* src_img, const std::vector<PredictResult>& predict_result) {
    (void)src_img;
    (void)predict_result;
}
