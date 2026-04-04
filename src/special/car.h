#pragma once

#include <vector>

#include "../include/common_types.h"


constexpr int CLASS_ID_CAR = 1;

class Car {
public:
    Car();
    Car(Config &config);

    enum  CarFlag {
    CAR_NONE = 0,
    CAR_AVOID = 1,
    CAR_LOST = 2,
};
    void check_car(const std::vector<PredictResult>& predict_result);
    void run_car(const void* src_img, const std::vector<PredictResult>& predict_result);

    CarFlag flag_car;


private:
    
};
