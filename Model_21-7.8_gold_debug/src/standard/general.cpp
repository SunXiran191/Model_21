#include "general.hpp"

#include <cmath>
#include <iostream>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

General::General() : counter_(0), last_error_(0.0) {}

void General::save_picture(const cv::Mat &image, int delta, const std::string &prefix)
{
    counter_ += delta;
    std::cout << "image:" << counter_ << "\n";
    const std::string name = save_dir_ + std::to_string(counter_) + prefix + ".jpg";
    cv::imwrite(name, image);
}

int General::factorial(int x) const
{
    int f = 1;
    for (int i = 1; i <= x; ++i)
    {
        f *= i;
    }
    return f;
}

std::vector<cv::Point2d> General::bezier(double dt,
                                         const std::vector<cv::Point2d> &input_points) const
{
    std::vector<cv::Point2d> output;
    if (input_points.empty() || dt <= 0.0)
    {
        return output;
    }

    for (double t = 0.0; t <= 1.0 + 1e-9; t += dt)
    {
        std::vector<cv::Point2d> tmp = input_points;
        for (std::size_t k = tmp.size() - 1; k > 0; --k)
        {
            for (std::size_t i = 0; i < k; ++i)
            {
                tmp[i].x = (1.0 - t) * tmp[i].x + t * tmp[i + 1].x;
                tmp[i].y = (1.0 - t) * tmp[i].y + t * tmp[i + 1].y;
            }
        }
        output.push_back(tmp.front());
    }
    return output;
}

double General::sigma(const std::vector<double> &vec, int n, int m) const
{
    if (vec.empty())
    {
        return 0.0;
    }
    const int start = std::max(0, n);
    const int end = std::min(static_cast<int>(vec.size()) - 1, m);
    if (start > end)
    {
        return 0.0;
    }

    double s = 0.0;
    for (int i = start; i <= end; ++i)
    {
        s += vec[static_cast<std::size_t>(i)];
    }
    return s;
}

double General::filter(double value)
{
    constexpr std::size_t kWindowSize = 5;
    filter_window_.push_back(value);
    if (filter_window_.size() > kWindowSize)
    {
        filter_window_.pop_front();
    }

    double sum = 0.0;
    for (double v : filter_window_)
    {
        sum += v;
    }
    return sum / static_cast<double>(filter_window_.size());
}

double General::pid_realize_a(double actual, double set_val, double _p, double _d)
{
    const double error = set_val - actual;
    const double out = _p * error + _d * (error - last_error_);
    last_error_ = error;
    return out;
}