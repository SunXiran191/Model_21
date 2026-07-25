#pragma once

#include <deque>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

#include "../../res/configs/param.hpp"
#include "../../res/include/aiget.hpp"
#include "../common/utils.hpp"

using POINT = cv::Point;

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <numeric>

class General
{
public:
    General();

    int clip(int x, int low, int up) const;
    void set_save_dir(const std::string &dir) { save_dir_ = dir; }
    void save_picture(const cv::Mat &image, int delta = 1, const std::string &prefix = "");
    int factorial(int x) const;
    std::vector<cv::Point2d> bezier(double dt, const std::vector<cv::Point2d> &input_points) const;
    double sigma(const std::vector<double> &vec, int n, int m) const;
    double filter(double value);
    double pid_realize_a(double actual, double set_val, double _p, double _d);
    // cv::Point2d transf(int i, int j) const;
    // cv::Point2d reverse_transf(int i, int j) const;

private:
    cv::Mat rotation_;
    int counter_;
    std::deque<double> filter_window_;
    double last_error_;
    std::string save_dir_ = "../image/";  // 图片保存目录，可通过 set_save_dir 配置
};
