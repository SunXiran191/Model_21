#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include "../common/utils.hpp"
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <string.h>

class LineTracker
{
public:
    // 传统cv提取图像中的蓝色箭头中心点
    std::vector<cv::Point> ExtractArrows_CV(const cv::Mat &src, cv::Mat &debug_mask);

    // ai语义分割提取图像中的蓝色箭头中心点
    std::vector<cv::Point> ExtractArrows_AI(const cv::Mat &src, cv::Mat &debug_mask);

};
