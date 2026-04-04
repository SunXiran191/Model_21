#include "human.h"

using namespace std;
using namespace cv;

Human::Human(){}
Human::Human(Config &config){

}

void Human::check_human(const std::vector<PredictResult>& predict_result) {
    bool is_human_detected = false;

    for (const auto& target : predict_result) {
        if (target.class_id == human) {
            is_human_detected = true;
            break;
        }
    }

    if (is_human_detected) {
        flag_human = HumanFlag::HUMAN_DETECT;
    } else {
        if (flag_human == HumanFlag::HUMAN_DETECT) {
            flag_human = HumanFlag::HUMAN_LOST;
        } else if (flag_human == HumanFlag::HUMAN_LOST) {
            flag_human = HumanFlag::HUMAN_NONE;
        }
    }

}