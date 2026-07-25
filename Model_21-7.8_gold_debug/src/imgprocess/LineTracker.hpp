#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include "../common/utils.hpp"
#include "../special/branch.hpp"
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <string.h>

struct Config;

class LineTracker
{
public:
    LineTracker();
    void init(const Config &config);

    // 传统cv提取图像中的蓝色箭头中心点
    std::vector<cv::Point> ExtractArrows_CV(const cv::Mat &src, cv::Mat &debug_mask);

    // 语义分割提取赛道中心点
    void ExtractArrows_AI(const cv::Mat &src,
                          cv::Mat &debug_mask,
                          std::vector<cv::Point> &path_straight,
                          std::vector<cv::Point> &path_branch);

    // 等距采样
    void resample_points(const std::vector<cv::Point> &input,
                         int input_size,
                         std::vector<cv::Point> &output,
                         int &output_size,
                         float dist_threshold);

    // 中线拟合偏置参数（由 Config 初始化）
    int track_width_left;
    int track_width_right;

    // standard.cpp可调用 check_branch / run_branch
    Branch branch_;
};
