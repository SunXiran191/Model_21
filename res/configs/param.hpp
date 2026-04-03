#pragma once

#include <string>
#include "../include/json.hpp"
struct Config{
    std::string jsonpath = "$.config";
    std::string model_path = "./model";

    bool enAI = true;



NLOHMANN_DEFINE_TYPE_INTRUSIVE(Config, jsonPth,wifi_start,ttyUsb, en_show, en_AI_show,speedLow, speedHigh, speedBridge,speedRing, speedCross,speedDown,aim_distance_far,aim_distance_near,aim_distance_ring,
                                       ring_entering_p_k,ring_p_k,steering_p,steering_d,curve_p,curve_d, ring_d,exposure,temprature,brightness, debug, saveImg, enAI,
                                       elem_order,speed_order,aim_dis_n_order,aim_angle_p_order,aim_angle_d_order,obstacle_order_index, model, video, score,CATERING_STOP_TIMESTAMP,
                                        CHARGING_LEFT_FIRST_GOIN_TIME,CHARGING_LEFT_SECOND_GOIN_TIME,CHARGING_RIGHT_FIRST_GOIN_TIME,CHARGING_RIGHT_SECOND_GOIN_TIME,CHARGING_LEFT_FIRST_OUT_TIME_ADD,CHARGING_LEFT_SECOND_OUT_TIME_ADD,CHARGING_RIGHT_FIRST_OUT_TIME_ADD,CHARGING_RIGHT_SECOND_OUT_TIME_ADD,CHARGING_GOIN_AIM_SPEED,
                                        STOP_EXPECT_Y,LAYBY_SHIFT_LEFT,LAYBY_SHIFT_RIGHT,LAYBY_PARKING_TIMER,LAYBY_AIM_SPEED,OBSTACLE_width_conical,OBSTACLE_width_pedestrian,OBSTACLE_width_block,block_aim_speed,cone_aim_speed,
                                        cone_p,cone_d,pe_p,pe_d,block_p,block_d
                                        );

}