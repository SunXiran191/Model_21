#pragma once

#include <deque>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

class General {
public:
    General();

    int clip(int x, int low, int up) const;
    void save_picture(const cv::Mat& image, int delta = 1, const std::string& prefix = "");
    int factorial(int x) const;
    std::vector<cv::Point2d> bezier(double dt, const std::vector<cv::Point2d>& input_points) const;
    double sigma(const std::vector<double>& vec, int n, int m) const;
    double filter(double value);
    double pid_realize_a(double actual, double set_val, double _p, double _d);
    cv::Point2d transf(int i, int j) const;
    cv::Point2d reverse_transf(int i, int j) const;

private:
    cv::Mat rotation_;
    cv::Mat change_un_mat_;
    cv::Mat re_change_un_mat_;
    int counter_;
    std::deque<double> filter_window_;
    double last_error_;
};
