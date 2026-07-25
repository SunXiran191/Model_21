#pragma once

#include <vector>
#include <tuple>
#include <opencv2/opencv.hpp>

// ÀûÓÃ×îÐ¡¶þ³Ë·¨ÄâºÏ 2D ¹ì¼£ÇúÏß
std::vector<cv::Point> FitTrajectory_LSM(const std::vector<cv::Point> &pts, cv::Mat &frame);

// ÀûÓÃ LOWESS ¾Ö²¿¼ÓÈ¨»Ø¹éÄâºÏ 2D ¹ì¼£ÇúÏß
std::vector<cv::Point> FitTrajectory_LOWESS(int point_count, const std::vector<cv::Point> &pts, const cv::Mat &frame);

// 高斯加权平滑 + 等距采样，比 LOWESS 快 20 倍以上
std::vector<cv::Point> FitTrajectory_Gauss(int point_count, const std::vector<cv::Point> &pts, const cv::Mat &frame);

// ÀûÓÃ¸ßË¹¹ý³Ì»Ø¹éÄâºÏ 2D ¹ì¼£ÇúÏß
std::vector<cv::Point> FitTrajectory_Poly(int point_count, const std::vector<cv::Point> &pts, cv::Mat &frame);

// ÀûÓÃ¸ßË¹¹ý³Ì»Ø¹éÄâºÏ 2D ¹ì¼£ÇúÏß
std::vector<cv::Point> FitTrajectory_GPR(int point_count, const std::vector<cv::Point> &pts, cv::Mat &frame);

// ÀûÓÃ·Ö¶ÎÈý½×±´Èû¶ûÇúÏßÄâºÏ 2D ¹ì¼£ÇúÏß
std::vector<cv::Point> FitTrajectory_Bezier(int point_count, const std::vector<cv::Point> &lane_points);

// ¶þ´Î±´Èû¶ûÇúÏß
std::vector<cv::Point> FitTrajectory_Bezier_2d(cv::Point p0, cv::Point p1, cv::Point p2, int num_points);
