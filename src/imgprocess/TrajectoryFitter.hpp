#pragma once

#include <vector>
#include <tuple>
#include <opencv2/opencv.hpp>

class TrajectoryFitter
{
public:
    // 提取图像中的蓝色箭头中心点
    std::vector<cv::Point> ExtractArrows(const cv::Mat &src, cv::Mat &debug_mask);

    // 利用最小二乘法拟合 2D 轨迹曲线
    std::vector<cv::Point> FitTrajectory_LSM(const std::vector<cv::Point> &pts, cv::Mat &frame);

    // 利用 LOWESS 局部加权回归拟合 2D 轨迹曲线
    std::vector<cv::Point> FitTrajectory_LOWESS(int point_count, const std::vector<cv::Point> &pts, cv::Mat &frame);

    // 利用高斯过程回归拟合 2D 轨迹曲线
    std::vector<cv::Point> FitTrajectory_Poly(int point_count, const std::vector<cv::Point> &pts, cv::Mat &frame);

    // 利用高斯过程回归拟合 2D 轨迹曲线
    std::vector<cv::Point> FitTrajectory_GPR(int point_count, const std::vector<cv::Point> &pts, cv::Mat &frame);

    // 利用分段三阶贝塞尔曲线拟合 2D 轨迹曲线
    std::vector<cv::Point> FitTrajectory_Bezier(int point_count, const std::vector<cv::Point> &lane_points);
};
