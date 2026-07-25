#include "LineTracker.hpp"
#include "../imgprocess/transform.hpp"
#include "../../res/configs/param.hpp"
#include <algorithm>
#include <iostream>
#include <cmath>

using namespace std;
using namespace cv;

LineTracker::LineTracker()
    : track_width_left(130),
      track_width_right(130)
{
}

void LineTracker::init(const Config &config)
{
    track_width_left = config.track_width_left;
    track_width_right = config.track_width_right;
}

vector<Point> LineTracker::ExtractArrows_CV(const Mat &src, Mat &debug_mask)
{
    if (src.empty())
    {
        debug_mask.release();
        return {};
    }

    //// ROI 裁剪提速
    // int roi_offset_y = src.rows * 0.6;
    // Rect roi(0, roi_offset_y, src.cols, src.rows - roi_offset_y);
    // Mat src_roi = src(roi); // 截取 ROI 区域

    Mat hsv;
    cvtColor(src, hsv, COLOR_BGR2HSV);

    // HSV阈值
    Scalar lower_blue(85, 100, 100);
    Scalar upper_blue(125, 255, 255);

    inRange(hsv, lower_blue, upper_blue, debug_mask);

    // 形态学操作
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(debug_mask, debug_mask, MORPH_OPEN, kernel);
    morphologyEx(debug_mask, debug_mask, MORPH_CLOSE, kernel); // 等会儿注释了看看效果

    int nwindows = 15;                // 滑窗数量
    int window_height = 20;           // 滑窗高度
    int margin = debug_mask.cols / 4; // 滑窗半宽
    int minpix = 20;                  // 最小像素数，低于该值认为赛道丢失

    // 寻找起始点
    Mat bottom_half = debug_mask(Rect(0, debug_mask.rows / 10, debug_mask.cols, debug_mask.rows / 10));
    Mat histogram;
    reduce(bottom_half, histogram, 0, REDUCE_SUM, CV_32S);

    Point max_loc;
    minMaxLoc(histogram, NULL, NULL, NULL, &max_loc);
    int current_x = max_loc.x;

    // 默认值画面中间
    if (histogram.at<int>(0, current_x) == 0)
    {
        current_x = debug_mask.cols / 2;
    }

    int dx = 0;                // 上一帧的横向位移
    int consecutive_lost = 0;  // 连续丢失赛道的窗口数
    vector<Point> lane_points; // 赛道中心点

    // 向上滑窗寻线
    for (int w = 0; w < nwindows; w++)
    {
        // 计算当前窗口的边界
        int win_y_low = debug_mask.rows - (w + 1) * window_height;
        int win_y_high = debug_mask.rows - w * window_height;
        int win_x_low = clip(current_x - margin, 0, debug_mask.cols - 1);
        int win_x_high = clip(current_x + margin, 0, debug_mask.cols - 1);

        // 提取当前窗口内的图像ROI
        Mat window_roi = debug_mask(Rect(win_x_low, win_y_low, win_x_high - win_x_low, window_height));

        vector<Point> nonZeroLocations;
        findNonZero(window_roi, nonZeroLocations);
        int found_pixels = nonZeroLocations.size();

        Scalar box_color = Scalar(0, 0, 255);

        // 判断是否被遮挡
        if (found_pixels > minpix)
        {
            // 找到足够的赛道像素，没有被遮挡
            int sum_x = 0;
            for (const auto &pt : nonZeroLocations)
            {
                sum_x += pt.x;
            }
            int mean_x = sum_x / found_pixels + win_x_low;

            // 更新惯性趋势
            dx = mean_x - current_x;
            current_x = mean_x;
            consecutive_lost = 0;
            box_color = Scalar(0, 255, 0);

            lane_points.push_back(Point(current_x, win_y_low + window_height / 2));
        }
        else
        {
            // 像素不够，启动“惯性”预测
            current_x = current_x + dx;
            consecutive_lost++;

            if (consecutive_lost > 1)
            {
                break;
            }

            lane_points.push_back(Point(clip(current_x, 0, debug_mask.cols - 1), win_y_low + window_height / 2));
        }
    }
    return lane_points;
}

void LineTracker::ExtractArrows_AI(const Mat &src, Mat &debug_mask, vector<Point> &path_straight,
                                   vector<Point> &path_branch)
{
    // 清空输出状态
    path_straight.clear();
    path_branch.clear();

    branch_.reset_frame();

    if (src.empty())
    {
        debug_mask.release();
        return;
    }

    // 预处理
    if (src.channels() == 1)
    {
        debug_mask = src;
    }
    else
    {
        cvtColor(src, debug_mask, COLOR_BGR2GRAY);
    }
    threshold(debug_mask, debug_mask, 1, 255, THRESH_BINARY);

    // 保留最大连通域
    vector<vector<Point>> all_contours;
    findContours(debug_mask, all_contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    if (all_contours.empty())
        return;

    int best_idx = 0;
    double best_area = -1.0;
    for (size_t i = 0; i < all_contours.size(); ++i)
    {
        double area = contourArea(all_contours[i]);
        if (area > best_area)
        {
            best_area = area;
            best_idx = static_cast<int>(i);
        }
    }
    debug_mask = Mat::zeros(debug_mask.size(), CV_8UC1);
    drawContours(debug_mask, all_contours, best_idx, Scalar(255), FILLED);

    // IPM 逆透视变换
    Mat ipm_mask;
    {
        Mat M(3, 3, CV_64F, (void *)change_un_Mat);

        warpPerspective(debug_mask, ipm_mask, M, debug_mask.size(), INTER_LINEAR); // 线性插值

        threshold(ipm_mask, ipm_mask, 127, 255, THRESH_BINARY);
    }

    if (countNonZero(ipm_mask) < 50)
    {
        return; // IPM变换后几乎无有效像素，回退：直接输出空路径
    }

    int ipm_h = ipm_mask.rows;
    int ipm_w = ipm_mask.cols;

    debug_mask = ipm_mask.clone();

    // 从左向右拟合主路
    // 从右向左拟合岔路
    int last_left_x = ipm_w / 2;
    int last_right_x = ipm_w / 2 + track_width_right / 2;
    int pix_missing = 0;
    const int MAX_MISSING = 5;

    // 存储每行边缘坐标，供 Branch::detect_t_junction 使用
    std::vector<std::pair<int, int>> row_edges; // (left_x, right_x)
    row_edges.reserve(ipm_h);

    for (int h = ipm_h - 1; h >= 0; --h)
    {
        const uchar *row_ptr = ipm_mask.ptr<uchar>(h);

        // 左锚点：找最左白像素，向右延伸track_width_left
        int left_x = -1;
        for (int w = 0; w < ipm_w; ++w)
        {
            if (row_ptr[w] > 0)
            {
                left_x = w;
                break;
            }
        }

        // 右锚点：找最右白像素，向左延伸track_width_right
        int right_x = -1;
        for (int w = ipm_w - 1; w >= 0; --w)
        {
            if (row_ptr[w] > 0)
            {
                right_x = w;
                break;
            }
        }

        // 记录该行边缘坐标（供 T 字路口检测）
        row_edges.push_back(std::make_pair(left_x, right_x));

        // 该行无白色像素，使用上一行的预测值
        if (left_x < 0 || right_x < 0)
        {
            pix_missing++;
            if (pix_missing > MAX_MISSING)
                continue;
            if (left_x < 0)
                left_x = last_left_x;
            if (right_x < 0)
                right_x = last_right_x;
        }
        else
        {
            pix_missing = 0;
            last_left_x = left_x;
            last_right_x = right_x;
        }

        // 左锚点 ROI: [left_x, left_x + track_width_left]
        int roi_l_left = left_x;
        int roi_l_right = min(left_x + track_width_left, ipm_w - 1);
        int sum_l = 0, cnt_l = 0;
        for (int w = roi_l_left; w <= roi_l_right; ++w)
        {
            if (row_ptr[w] > 0)
            {
                sum_l += w;
                cnt_l++;
            }
        }
        int center_l = (cnt_l > 0) ? sum_l / cnt_l : roi_l_left + track_width_left / 2;
        path_straight.push_back(Point(center_l, h));

        // 右锚点 ROI: [right_x - track_width_right, right_x]
        int roi_r_left = max(right_x - track_width_right, 0);
        int roi_r_right = right_x;
        int sum_r = 0, cnt_r = 0;
        for (int w = roi_r_left; w <= roi_r_right; ++w)
        {
            if (row_ptr[w] > 0)
            {
                sum_r += w;
                cnt_r++;
            }
        }
        int center_r = (cnt_r > 0) ? sum_r / cnt_r : roi_r_right - track_width_right / 2;
        path_branch.push_back(Point(center_r, h));

        // 岔路：左ROI右侧多余的白色像素
        int branch_sum_x = 0, branch_cnt = 0;
        for (int w = roi_l_right + 1; w < ipm_w; ++w)
        {
            if (row_ptr[w] > 0)
            {
                branch_sum_x += w;
                branch_cnt++;
            }
        }
        int branch_center = (branch_cnt >= branch_.branch_min_pixels)
                                ? branch_sum_x / branch_cnt
                                : -1;
        if (branch_cnt >= branch_.branch_min_pixels)
        {
            Branch::BranchRow br;
            br.row = h;
            br.pixel_count = branch_cnt;
            br.center_x = branch_center;
            br.left_edge_x = left_x;
            branch_.branch_rows_.push_back(br);
        }
    }

    // T 字路口检测：基于 IPM 边缘坐标
    branch_.detect_t_junction(row_edges, (int)row_edges.size());
}

void LineTracker::resample_points(const vector<cv::Point> &input, int input_size, vector<cv::Point> &output, int &output_size, float dist_threshold)
{
    output.clear();

    if (dist_threshold < 2.0f)
    {
        dist_threshold = 4.0f;
    }

    if (input_size < 2)
    {
        output = input;
        output_size = output.size();
        return;
    }

    output.push_back(input[0]);
    float current_dist = 0.0f;
    float target_dist = dist_threshold;

    for (int i = 1; i < input_size; ++i)
    {
        float dx = input[i].x - input[i - 1].x;
        float dy = input[i].y - input[i - 1].y;
        float segment_len = std::sqrt(dx * dx + dy * dy);

        while (current_dist + segment_len >= target_dist)
        {
            float ratio = (target_dist - current_dist) / segment_len;

            float nx = input[i - 1].x + dx * ratio;
            float ny = input[i - 1].y + dy * ratio;

            output.push_back(cv::Point(cvRound(nx), cvRound(ny)));

            target_dist += dist_threshold;
        }
        current_dist += segment_len;
    }
    output_size = output.size();
}
