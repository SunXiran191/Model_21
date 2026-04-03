#include "car.h"



Car::Car(){}


void Car::check_car(const std::vector<DetectionTarget>& predict_result) {
    bool is_car_detected = false;

    for (const auto& target : predict_result) {
        if (target.class_id == CLASS_ID_CAR) {
            is_car_detected = true;
            break;
        }
    }

    if (is_car_detected) {
        flag_car = CarFlag::CAR_AVOID;
    } else {
        if (flag_car == CarFlag::CAR_AVOID) {
            flag_car = CarFlag::CAR_LOST;
        } else if (flag_car == CarFlag::CAR_LOST) {
            flag_car = CarFlag::CAR_NONE;
        }
    }
}

void Car::run_car(const void* src_img, const std::vector<DetectionTarget>& predict_result) {
    (void)src_img;
    (void)predict_result;
}
