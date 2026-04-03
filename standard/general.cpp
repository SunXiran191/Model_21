#include "general.h"

#include <cmath>
#include <iostream>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

General::General() : counter_(0), last_error_(0.0) {
    std::vector<cv::Point2f> dst_points = {
        {123.0F, 105.0F}, {195.0F, 104.0F}, {102.0F, 135.0F}, {211.0F, 134.0F}};
    std::vector<cv::Point2f> src_points = {
        {110.0F, 40.0F}, {210.0F, 40.0F}, {110.0F, 140.0F}, {210.0F, 140.0F}};

    rotation_ = cv::getPerspectiveTransform(src_points, dst_points);

    change_un_mat_ = (cv::Mat_<double>(3, 3) << -3.614457831325295, -5.466867469879507,
                      744.9397590361433, -9.964747281289631e-16, -11.48343373493974,
                      999.0963855421669, -4.864999171266168e-18, -0.03463855421686741, 1.0);

    re_change_un_mat_ =
        (cv::Mat_<double>(3, 3) << 0.5571147540983606, -0.4899672131147541, 74.50754098360656,
         4.174901165517516e-17, -0.08708196721311479, 87.00327868852459,
         1.719608593143229e-19, -0.003016393442622951, 1.0);
}

int General::clip(int x, int low, int up) const {
    return x > up ? up : (x < low ? low : x);
}

void General::save_picture(const cv::Mat& image, int delta, const std::string& prefix) {
    counter_ += delta;
    std::cout << "image:" << counter_ << "\n";
    const std::string img_path = "/home/edgeboard/Run_ACCM_2025Demo/image/";
    const std::string name = img_path + std::to_string(counter_) + prefix + ".jpg";
    cv::imwrite(name, image);
}

int General::factorial(int x) const {
    int f = 1;
    for (int i = 1; i <= x; ++i) {
        f *= i;
    }
    return f;
}

std::vector<cv::Point2d> General::bezier(double dt,
                                         const std::vector<cv::Point2d>& input_points) const {
    std::vector<cv::Point2d> output;
    if (input_points.empty() || dt <= 0.0) {
        return output;
    }

    for (double t = 0.0; t <= 1.0 + 1e-9; t += dt) {
        std::vector<cv::Point2d> tmp = input_points;
        for (std::size_t k = tmp.size() - 1; k > 0; --k) {
            for (std::size_t i = 0; i < k; ++i) {
                tmp[i].x = (1.0 - t) * tmp[i].x + t * tmp[i + 1].x;
                tmp[i].y = (1.0 - t) * tmp[i].y + t * tmp[i + 1].y;
            }
        }
        output.push_back(tmp.front());
    }
    return output;
}

double General::sigma(const std::vector<double>& vec, int n, int m) const {
    if (vec.empty()) {
        return 0.0;
    }
    const int start = std::max(0, n);
    const int end = std::min(static_cast<int>(vec.size()) - 1, m);
    if (start > end) {
        return 0.0;
    }

    double s = 0.0;
    for (int i = start; i <= end; ++i) {
        s += vec[static_cast<std::size_t>(i)];
    }
    return s;
}

double General::filter(double value) {
    constexpr std::size_t kWindowSize = 5;
    filter_window_.push_back(value);
    if (filter_window_.size() > kWindowSize) {
        filter_window_.pop_front();
    }

    double sum = 0.0;
    for (double v : filter_window_) {
        sum += v;
    }
    return sum / static_cast<double>(filter_window_.size());
}

double General::pid_realize_a(double actual, double set_val, double _p, double _d) {
    const double error = set_val - actual;
    const double out = _p * error + _d * (error - last_error_);
    last_error_ = error;
    return out;
}

cv::Point2d General::transf(int i, int j) const {
    cv::Mat src = (cv::Mat_<double>(3, 1) << static_cast<double>(i), static_cast<double>(j), 1.0);
    cv::Mat dst = change_un_mat_ * src;
    const double w = dst.at<double>(2, 0);
    if (std::abs(w) < 1e-12) {
        return cv::Point2d(0.0, 0.0);
    }
    return cv::Point2d(dst.at<double>(0, 0) / w, dst.at<double>(1, 0) / w);
}

cv::Point2d General::reverse_transf(int i, int j) const {
    cv::Mat src = (cv::Mat_<double>(3, 1) << static_cast<double>(i), static_cast<double>(j), 1.0);
    cv::Mat dst = re_change_un_mat_ * src;
    const double w = dst.at<double>(2, 0);
    if (std::abs(w) < 1e-12) {
        return cv::Point2d(0.0, 0.0);
    }
    return cv::Point2d(dst.at<double>(0, 0) / w, dst.at<double>(1, 0) / w);
}
