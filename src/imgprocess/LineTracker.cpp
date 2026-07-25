#include "LineTracker.hpp"
#include <algorithm>
#include <iostream>

using namespace std;
using namespace cv;

vector<Point> LineTracker::ExtractArrows_CV(const Mat &src, Mat &debug_mask)
{
    //// ROI 裁剪提速 (大幅提升 FPS)
    // int roi_offset_y = src.rows * 0.6;
    // Rect roi(0, roi_offset_y, src.cols, src.rows - roi_offset_y);
    // Mat src_roi = src(roi); // 截取 ROI 区域 (浅拷贝，极快)

    Mat hsv;
    cvtColor(src, hsv, COLOR_BGR2HSV);

    // 蓝色/青色的 HSV 阈值
    Scalar lower_blue(85, 100, 100);
    Scalar upper_blue(125, 255, 255);

    //// 为了让主程序的 debug_mask 显示大小一致，我们创建一个全黑的全尺寸图像，只在 ROI 区域写入结果
    // debug_mask = Mat::zeros(src.size(), CV_8UC1);
    // Mat mask_roi = debug_mask(roi);
    inRange(hsv, lower_blue, upper_blue, debug_mask);

    // 形态学操作（只在 ROI 区域内执行，省时）
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(debug_mask, debug_mask, MORPH_OPEN, kernel);
    morphologyEx(debug_mask, debug_mask, MORPH_CLOSE, kernel); // 等会儿注释了看看效果

    // ==========================================
    // 第二步：初始化“惯性”滑窗参数
    // ==========================================
    int nwindows = 15;                // 滑窗的数量 (从下到上)
    int window_height = 20;           // 每个滑窗的高度
    int margin = debug_mask.cols / 4; // 滑窗的半宽 (宽 = 320)
    int minpix = 20;                  // 窗口内最少需要多少个白点才算有效找到赛道

    // 寻找起始点：统计图像下半部分的像素直方图，找到底部赛道的初始X坐标
    Mat bottom_half = debug_mask(Rect(0, debug_mask.rows / 10, debug_mask.cols, debug_mask.rows / 10));
    Mat histogram;
    reduce(bottom_half, histogram, 0, REDUCE_SUM, CV_32S);

    Point max_loc;
    minMaxLoc(histogram, NULL, NULL, NULL, &max_loc);
    int current_x = max_loc.x; // 滑窗初始的X坐标

    // 如果底部一点蓝色都没有（起步就被挡住），给个默认值（画面中间）
    if (histogram.at<int>(0, current_x) == 0)
    {
        current_x = debug_mask.cols / 2;
    }

    // “惯性”参数初始化
    int dx = 0;                // 记录上一帧的横向位移 (斜率趋势)
    int consecutive_lost = 0;  // 连续丢失赛道的窗口数
    vector<Point> lane_points; // 存储提取出的赛道中心点

    // ==========================================
    // 第三步：带有惯性预测的向上滑窗寻线
    // ==========================================
    for (int w = 0; w < nwindows; w++)
    {
        // 计算当前窗口的边界 (注意不能越界)
        int win_y_low = debug_mask.rows - (w + 1) * window_height;
        int win_y_high = debug_mask.rows - w * window_height;
        int win_x_low = clamp(current_x - margin, 0, debug_mask.cols - 1);
        int win_x_high = clamp(current_x + margin, 0, debug_mask.cols - 1);

        // 提取当前窗口内的图像 ROI
        Mat window_roi = debug_mask(Rect(win_x_low, win_y_low, win_x_high - win_x_low, window_height));

        // 查找窗口内非零像素 (白点)
        vector<Point> nonZeroLocations;
        findNonZero(window_roi, nonZeroLocations);
        int found_pixels = nonZeroLocations.size();

        // 绘制当前矩形框 (红色：预测/盲找框，绿色：实际找到赛道的框)
        Scalar box_color = Scalar(0, 0, 255); // 默认红色

        // 核心逻辑：判断是否被遮挡
        if (found_pixels > minpix)
        {
            // 情况 A：找到足够的赛道像素，没有被遮挡
            int sum_x = 0;
            for (const auto &pt : nonZeroLocations)
            {
                sum_x += pt.x; // pt.x 是相对于 ROI 的坐标
            }
            int mean_x = sum_x / found_pixels + win_x_low; // 转换回原图坐标系

            // 更新惯性趋势 (当前中心点 - 上一个中心点)
            dx = mean_x - current_x;

            // 更新当前X坐标
            current_x = mean_x;
            consecutive_lost = 0;          // 重置丢失计数
            box_color = Scalar(0, 255, 0); // 绿色表示有效

            lane_points.push_back(Point(current_x, win_y_low + window_height / 2));
        }
        else
        {
            // 情况 B：像素不够！前方是行人、黄车或者断线区 (发生遮挡)
            // 启动“惯性”预测：利用之前的趋势 dx，推测下一次赛道应该出现的位置
            current_x = current_x + dx;
            consecutive_lost++;

            // 如果连续丢失太多次(比如连续5个框都没找到)，说明可能真丢线了，或者处于急弯
            if (consecutive_lost > 1)
            {
                break; // 直接跳出 for 循环，停止向上滑窗
            }

            // 虽然是盲猜的，但为了保持路径连续，依然把预测点加入路径
            lane_points.push_back(Point(clamp(current_x, 0, debug_mask.cols - 1), win_y_low + window_height / 2));
        }

        //// 画出滑窗，方便调试观察
        // rectangle(out_img, Point(win_x_low, win_y_low), Point(win_x_high, win_y_high), box_color, 2);
    }

    return lane_points;
}

vector<Point> LineTracker::ExtractArrows_AI(const Mat &src, Mat &debug_mask)
{
    
}
