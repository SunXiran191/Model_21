#include "car.h"

// 赛道上有金币、行人、车等障碍。金币需要收集，行人需要避免，车需要避免。我现在的想法是，对于金币，接收模型推理结果的金币置信度和坐标，置信度大于0.8则取金币中心坐标，与金币y坐标距离前后各自第五个像素的赛道线坐标作贝塞尔曲线拟合，作为赛道巡线结果。

using namespace std;
using namespace cv;

Car::Car() {}
Car::Car(Config &config) {}

void Car::check_car(const std::vector<PredictResult> &predict_result)
{
    bool is_car_detected = false;
    static int find_cnt = 0;
    static int lost_cnt = 0;

    for (const auto &target : predict_result)
    {
        if (target.class_id == CLASS_ID_CAR)
        {
            if (target.score > PREDICT_THRESH)
            {
                is_car_detected = true;
                break;
            }
        }
    }

    if (car_flag == CarFlag::CAR_NONE)
    {
        if (is_car_detected)
        {
            find_cnt++;
            if (find_cnt >= PREDICT_THRESH)
            {
                car_flag = CarFlag::CAR_DETECTED;
                logger.info("CAR detected!");
                find_cnt = 0;
            }
        }
    }
    else if (car_flag == CarFlag::CAR_DETECTED)
    {
        if (!is_car_detected)
        {
            car_flag = CarFlag::CAR_LOST;
            lost_cnt++;
        }
    }
    else if (car_flag == CarFlag::CAR_LOST)
    {
        if (!is_car_detected && lost_cnt > LOST_THRESH)
        {
            car_flag = CarFlag::CAR_NONE;
            lost_cnt = 0;
        }
    }
}

void Car::run_car(const cv::Mat &src_img,
                  const std::vector<PredictResult> &predict_result,
                  const std::vector<POINT> &CV_pointsOrigin,
                  std::vector<POINT> &trackPoints,
                  int CV_pointsOrigin_size,
                  int trackPoints_size)
{
    found_valid_car = false;
    static int cnt_getcar = 0;
    static int cnt_lostcar = 0;

    if (car_flag == CarFlag::CAR_DETECTED)
    {

        int dist_forward = 40;
        int dist_backward = 40;
        int start_idx = -1;
        int end_idx = -1;

        for (int i = 0; i < CV_pointsOrigin_size; ++i)
        {
            if (start_idx == -1 && CV_pointsOrigin[i].y <= general.general.clipf(t_car_point.y + dist_forward, 0, src_img.rows - 1))
            {
                start_idx = i;
            }

            if (end_idx == -1 && CV_pointsOrigin[i].y <= general.general.clipf(t_car_point.y - dist_backward, 0, src_img.rows - 1))
            {
                end_idx = i;
                break;
            }
        }

        if (start_idx < 0 || end_idx < 0 || start_idx >= end_idx)
        {
            car_flag = CarFlag::CAR_LOCATED;
        }
    }
    else if (car_flag == CarFlag::CAR_LOCATED)
    {
        int img_center_x = src_img.cols / 2;
        int avoid_dir = (t_car_point.x < img_center_x) ? 1 : -1;
        int avoid_dist = 50; // 改全局变量

        POINT p_avoid = {t_car_point.x + avoid_dir * avoid_dist, t_car_point.y};

        POINT p_start = CV_pointsOrigin[start_idx];
        POINT p_end = CV_pointsOrigin[end_idx];
        std::vector<POINT> bezier_segment =
            TrajectoryFitter::FitTrajectory_Bezier_2d(p_start, p_avoid, p_end, 20);

        trackPoints.erase(trackPoints.begin() + start_idx, trackPoints.begin() + end_idx);
        trackPoints.insert(trackPoints.begin() + start_idx, bezier_segment.begin(), bezier_segment.end());

        logger.info("Car detected! Trajectory replanned.");

        if (t_car_point.y1 >= trackPoints.front().y)
        {
            cnt_avoidcar++;
        }
        if (cnt_avoidcar >= 10)
        {
            car_flag = CarFlag::CAR_AVOID;
            cnt_avoidcar = 0;
            logger.info("Car avoided!");
        }
    }
    else if (car_flag == CarFlag::CAR_AVOID)
    {
        for (const auto &target : predict_result)
        {
            if (!CLASS_ID_CAR) // 这是改过后的是否检测到车的布尔变量
            {
                cnt_lostcar++;
            }
        }
        if (cnt_lostcar >= 10)
        {
            car_flag = CarFlag::CAR_NONE;
            cnt_lostcar = 0;
            logger.info("Car none!");
        }
    }
}