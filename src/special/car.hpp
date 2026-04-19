#pragma once

#include <vector>
#include "../TrajectoryFitter.hpp"
#include "../standard/general.h"

constexpr int CLASS_ID_CAR = 1;

class Car
{
public:
    Car();
    Car(Config &config);

    enum CarFlag
    {
        CAR_NONE = 0,
        CAR_DETECTED = 1,
        CAR_AVOID = 2,
        CAR_LOST = 3,
    };
    CarFlag car_flag = CarFlag::CAR_NONE;

    void check_car(const std::vector<PredictResult> &predict_result);
    void Car::run_car(const cv::Mat &src_img, const std::vector<PredictResult> &predict_result, const std::vector<POINT> &CV_pointsOrigin, std::vector<POINT> &trackPoints, int CV_pointsOrigin_size, int trackPoints_size);

private:
};
