#pragma once

#include <string>
#include "../include/json.hpp"

struct Config{
    std::string jsonpath = "$.config";
    std::string model_path = "./model";

    bool enAI = true;
    bool enAI_show = true;

    float speedLow = 0.8f;
    float speedHigh = 1.5f;
    float speedavoid = 0.8f;



NLOHMANN_DEFINE_TYPE_INTRUSIVE(Config, jsonPth,wifi_start,ttyUsb, en_show, en_AI_show,speedLow, speedHigh, speedavoid
                                        );

}