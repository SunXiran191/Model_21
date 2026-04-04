#include "LineTracker.hpp"
#include <algorithm>
#include <iostream>

using namespace std;
using namespace cv;

vector<Point> LineTracker::ExtractArrows(const Mat& src, Mat& debug_mask) {
    //// ROI 裁剪提速 (大幅提升 FPS)
    //int roi_offset_y = src.rows * 0.6; 
    //Rect roi(0, roi_offset_y, src.cols, src.rows - roi_offset_y);
    //Mat src_roi = src(roi); // 截取 ROI 区域 (注意这只是浅拷贝，极快)
     
    Mat hsv;
    cvtColor(src, hsv, COLOR_BGR2HSV);

    // 蓝色/青色的 HSV 阈值
    Scalar lower_blue(85, 100, 100);
    Scalar upper_blue(125, 255, 255);

    //// 为了让主程序的 debug_mask 显示大小一致，我们创建一个全黑的全尺寸图像，只在 ROI 区域写入结果
    //debug_mask = Mat::zeros(src.size(), CV_8UC1);
    //Mat mask_roi = debug_mask(roi);
    inRange(hsv, lower_blue, upper_blue, debug_mask);

    // 形态学操作（只在 ROI 区域内执行，省时）
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(debug_mask, debug_mask, MORPH_OPEN, kernel);
    morphologyEx(debug_mask, debug_mask, MORPH_CLOSE, kernel);//等会儿注释了看看效果

    // ==========================================
// 第二步：初始化“惯性”滑窗参数
// ==========================================
    int nwindows = 15;                       // 滑窗的数量 (从下到上)
    int window_height = 20; // 每个滑窗的高度
    int margin = debug_mask.cols / 4;                         // 滑窗的半宽 (宽 = 320)
    int minpix = 20;                         // 窗口内最少需要多少个白点才算有效找到赛道

    // 寻找起始点：统计图像下半部分的像素直方图，找到底部赛道的初始X坐标
    Mat bottom_half = debug_mask(Rect(0, debug_mask.rows / 10, debug_mask.cols, debug_mask.rows / 10));
    Mat histogram;
    reduce(bottom_half, histogram, 0, REDUCE_SUM, CV_32S);

    Point max_loc;
    minMaxLoc(histogram, NULL, NULL, NULL, &max_loc);
    int current_x = max_loc.x; // 滑窗初始的X坐标

    // 如果底部一点蓝色都没有（起步就被挡住），给个默认值（画面中间）
    if (histogram.at<int>(0, current_x) == 0) {
        current_x = debug_mask.cols / 2;
    }

    // “惯性”参数初始化
    int dx = 0;              // 记录上一帧的横向位移 (斜率趋势)
    int consecutive_lost = 0;// 连续丢失赛道的窗口数
    vector<Point> lane_points; // 存储提取出的赛道中心点

    // ==========================================
    // 第三步：带有惯性预测的向上滑窗寻线
    // ==========================================
    for (int w = 0; w < nwindows; w++) {
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
        if (found_pixels > minpix) {
            // 情况 A：找到足够的赛道像素，没有被遮挡
            int sum_x = 0;
            for (const auto& pt : nonZeroLocations) {
                sum_x += pt.x; // pt.x 是相对于 ROI 的坐标
            }
            int mean_x = sum_x / found_pixels + win_x_low; // 转换回原图坐标系

            // 更新惯性趋势 (当前中心点 - 上一个中心点)
            dx = mean_x - current_x;

            // 更新当前X坐标
            current_x = mean_x;
            consecutive_lost = 0; // 重置丢失计数
            box_color = Scalar(0, 255, 0); // 绿色表示有效

            lane_points.push_back(Point(current_x, win_y_low + window_height / 2));
        }
        else {
            // 情况 B：像素不够！前方是行人、黄车或者断线区 (发生遮挡)
            // 启动“惯性”预测：利用之前的趋势 dx，推测下一次赛道应该出现的位置
            current_x = current_x + dx;
            consecutive_lost++;

            // 如果连续丢失太多次(比如连续5个框都没找到)，说明可能真丢线了，或者处于急弯
            if (consecutive_lost > 1) {
                break; // 直接跳出 for 循环，停止向上滑窗
            }

            // 虽然是盲猜的，但为了保持路径连续，依然把预测点加入路径
            lane_points.push_back(Point(clamp(current_x, 0, debug_mask.cols - 1), win_y_low + window_height / 2));
        }

        //// 画出滑窗，方便调试观察
        //rectangle(out_img, Point(win_x_low, win_y_low), Point(win_x_high, win_y_high), box_color, 2);
    }

    return lane_points;
}

vector<Point> LineTracker::FitTrajectory_LSM(const vector<Point>& pts, Mat& frame) {
    vector<Point> curve_pts;
    if (pts.size() < 3) return curve_pts; // 经过过滤后如果点数太少，就不强行拟合了

    int n = pts.size();
    Mat A(n, 3, CV_32F);
    Mat B(n, 1, CV_32F);

    for (int i = 0; i < n; ++i) {
        float y = static_cast<float>(pts[i].y);
        A.at<float>(i, 0) = y * y;
        A.at<float>(i, 1) = y;
        A.at<float>(i, 2) = 1.0f;
        B.at<float>(i, 0) = static_cast<float>(pts[i].x);
    }

    Mat coeffs;
    solve(A, B, coeffs, DECOMP_SVD);

    float a = coeffs.at<float>(0, 0);
    float b = coeffs.at<float>(1, 0);
    float c = coeffs.at<float>(2, 0);

    int y_start = pts.front().y;
    int y_end = pts.back().y;

    if (y_start <= y_end) return curve_pts;

    for (int y = y_start; y >= y_end; y -= 5) {
        int x = static_cast<int>(round(a * y * y + b * y + c));
        if (x >= 0 && x < frame.cols) {
            curve_pts.push_back(Point(x, y));
        }
    }
    return curve_pts;
}

//lowess局部加权回归拟合，适合处理噪点较多的情况，计算量较大
vector<Point> LineTracker::FitTrajectory_LOWESS(int point_count, const vector<Point>& pts, Mat& frame) {
    vector<Point> curve_pts;
    int n = pts.size();
    if (n < 2) return pts;

    // 平滑窗口大小 (占总点数的比例)
    double span = 0.4;
    int window_size = max(3, (int)(n * span));

    for (int i = 0; i < point_count; ++i) {
        double t = (double)i / (point_count - 1) * (n - 1); // 目标插值点位置

        double sum_w = 0, sum_wt = 0, sum_wt2 = 0, sum_wx = 0, sum_wy = 0, sum_wtx = 0, sum_wty = 0;

        for (int j = 0; j < n; ++j) {
            double diff = abs(t - j);
            if (diff < window_size) {
                // 三次方加权函数 (Tricube weight function)
                double rel_dist = diff / window_size;
                double w = pow(1.0 - pow(rel_dist, 3), 3);

                sum_w += w;
                sum_wt += w * j;
                sum_wt2 += w * j * j;
                sum_wx += w * pts[j].x;
                sum_wy += w * pts[j].y;
                sum_wtx += w * j * pts[j].x;
                sum_wty += w * j * pts[j].y;
            }
        }

        // 求解局部线性方程: [sum_w sum_wt; sum_wt sum_wt2] * [a; b] = [sum_wx; sum_wtx]
        double det = sum_w * sum_wt2 - sum_wt * sum_wt;
        if (abs(det) > 1e-6) {
            double x_fit = (sum_wt2 * sum_wx - sum_wt * sum_wtx) / det +
                (sum_w * sum_wtx - sum_wt * sum_wx) / det * t;
            double y_fit = (sum_wt2 * sum_wy - sum_wt * sum_wty) / det +
                (sum_w * sum_wty - sum_wt * sum_wy) / det * t;
            curve_pts.push_back(Point(cvRound(x_fit), cvRound(y_fit)));
        }
    }
    return curve_pts;
}

//多项式拟合，计算量小但对噪点敏感
vector<Point> LineTracker::FitTrajectory_Poly(int point_count, const vector<Point>& pts, Mat& frame) {
    vector<Point> curve_pts;
    int n = pts.size();
    int degree = 3; // 多项式阶数
    if (n <= degree) return pts;

    auto solvePoly = [&](const vector<double>& vals) {
        Mat A(n, degree + 1, CV_64F);
        Mat B(n, 1, CV_64F);
        for (int i = 0; i < n; i++) {
            double t = (double)i / (n - 1);
            for (int j = 0; j <= degree; j++) {
                A.at<double>(i, j) = pow(t, j);
            }
            B.at<double>(i, 0) = vals[i];
        }
        Mat coeffs;
        solve(A, B, coeffs, DECOMP_SVD);
        return coeffs;
        };

    vector<double> px, py;
    for (const auto& p : pts) { px.push_back(p.x); py.push_back(p.y); }

    Mat coeffX = solvePoly(px);
    Mat coeffY = solvePoly(py);

    for (int i = 0; i < point_count; i++) {
        double t = (double)i / (point_count - 1);
        double rx = 0, ry = 0;
        for (int j = 0; j <= degree; j++) {
            rx += coeffX.at<double>(j, 0) * pow(t, j);
            ry += coeffY.at<double>(j, 0) * pow(t, j);
        }
        curve_pts.push_back(Point(cvRound(rx), cvRound(ry)));
    }
    return curve_pts;
}

//高斯过程回归，适合处理复杂曲线，计算量较大
vector<Point> LineTracker::FitTrajectory_GPR(int point_count, const vector<Point>& pts, Mat& frame) {
    vector<Point> curve_pts;
    int n = pts.size();
    if (n < 2) return pts;

    // GPR 超参数
    double l = 0.6;      // 长度尺度 (Length scale)
    double sigma_f = 1.0; // 信号方差
    double sigma_n = 0.1; // 噪声标准差

    auto kernel = [&](double t1, double t2) {
        return sigma_f * exp(-pow(t1 - t2, 2) / (2 * l * l));
        };

    // 构建协方差矩阵 K
    Mat K(n, n, CV_64F);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double t_i = (double)i / (n - 1);
            double t_j = (double)j / (n - 1);
            K.at<double>(i, j) = kernel(t_i, t_j) + (i == j ? sigma_n * sigma_n : 0);
        }
    }

    Mat K_inv = K.inv(DECOMP_CHOLESKY);

    Mat Y_x(n, 1, CV_64F), Y_y(n, 1, CV_64F);
    for (int i = 0; i < n; i++) {
        Y_x.at<double>(i, 0) = pts[i].x;
        Y_y.at<double>(i, 0) = pts[i].y;
    }

    // 预测
    for (int i = 0; i < point_count; i++) {
        double t_star = (double)i / (point_count - 1);
        Mat k_star(1, n, CV_64F);
        for (int j = 0; j < n; j++) {
            k_star.at<double>(0, j) = kernel(t_star, (double)j / (n - 1));
        }

        Mat mu_x = k_star * K_inv * Y_x;
        Mat mu_y = k_star * K_inv * Y_y;

        curve_pts.push_back(Point(cvRound(mu_x.at<double>(0, 0)),
            cvRound(mu_y.at<double>(0, 0))));
    }
    return curve_pts;
}

//分段贝塞尔曲线
// 感觉不能用，很难排除噪点/极端点
vector<Point> LineTracker::FitTrajectory_Bezier(int point_count, const vector<Point>& lane_points)
{
    vector<Point> output;
    // 点数不足时直接返回原始点
    if (point_count < 2 || lane_points.empty() || lane_points.size() != point_count) {
        return lane_points;
    }

    // 平滑步长（控制拟合曲线的点数，值越小曲线越平滑）
    const double dt = 0.02; // 建议值：0.01~0.05，0.02兼顾平滑度和效率

    // 不同点数的适配逻辑
    // 1 只有2个点：线性拟合
    if (point_count == 2) {
        for (double t = 0; t <= 1 + 1e-6; t += dt) {
            int x = round((1 - t) * lane_points[0].x + t * lane_points[1].x);
            int y = round((1 - t) * lane_points[0].y + t * lane_points[1].y);
            output.push_back(Point(x, y));
        }
        return output;
    }

    // 2 只有3个点：二次贝塞尔拟合
    if (point_count == 3) {
        for (double t = 0; t <= 1 + 1e-6; t += dt) {
            double t1 = 1 - t;
            int x = round(t1 * t1 * lane_points[0].x + 2 * t1 * t * lane_points[1].x + t * t * lane_points[2].x);
            int y = round(t1 * t1 * lane_points[0].y + 2 * t1 * t * lane_points[1].y + t * t * lane_points[2].y);
            output.push_back(Point(x, y));
        }
        return output;
    }

    // 3 点数≥4：分段三次贝塞尔拟合
    // 辅助函数：计算单段三次贝塞尔点
    auto calcCubicBezier = [dt](const vector<Point>& ctrl_pts) -> vector<Point> {
        vector<Point> segment;
        for (double t = 0; t <= 1 + 1e-6; t += dt) {
            double t1 = 1 - t;
            double t1_3 = pow(t1, 3);
            double t1_2 = pow(t1, 2);
            double t_2 = pow(t, 2);
            double t_3 = pow(t, 3);

            int x = round(t1_3 * ctrl_pts[0].x + 3 * t1_2 * t * ctrl_pts[1].x +
                3 * t1 * t_2 * ctrl_pts[2].x + t_3 * ctrl_pts[3].x);
            int y = round(t1_3 * ctrl_pts[0].y + 3 * t1_2 * t * ctrl_pts[1].y +
                3 * t1 * t_2 * ctrl_pts[2].y + t_3 * ctrl_pts[3].y);
            segment.push_back(Point(x, y));
        }
        return segment;
        };

    // 每4个控制点为一段，重叠1个控制点
    for (size_t i = 0; i <= lane_points.size() - 4; i += 3) {
        // 提取当前段的4个控制点
        vector<Point> ctrl_pts(lane_points.begin() + i, lane_points.begin() + i + 4);
        vector<Point> segment_pts = calcCubicBezier(ctrl_pts);
        output.insert(output.end(), segment_pts.begin(), segment_pts.end());
    }

    // 处理最后不足4个点的剩余部分
    if (lane_points.size() % 3 != 1 && lane_points.size() > 4) {
        vector<Point> last_ctrl_pts(lane_points.end() - 4, lane_points.end());
        vector<Point> last_segment = calcCubicBezier(last_ctrl_pts);
        output.insert(output.end(), last_segment.begin(), last_segment.end());
    }

    return output;
}

//vector<Point> LineTracker::ExtractArrows(const Mat& src, Mat& debug_mask) {
//    // ROI 裁剪提速 (大幅提升 FPS)
//    int roi_offset_y = src.rows * 0.6;
//    Rect roi(0, roi_offset_y, src.cols, src.rows - roi_offset_y);
//    Mat src_roi = src(roi); // 截取 ROI 区域 (注意这只是浅拷贝，极快)
//
//    Mat hsv;
//    cvtColor(src_roi, hsv, COLOR_BGR2HSV);
//
//    // 蓝色/青色的 HSV 阈值
//    Scalar lower_blue(85, 100, 100);
//    Scalar upper_blue(125, 255, 255);
//
//    // 为了让主程序的 debug_mask 显示大小一致，我们创建一个全黑的全尺寸图像，只在 ROI 区域写入结果
//    debug_mask = Mat::zeros(src.size(), CV_8UC1);
//    Mat mask_roi = debug_mask(roi);
//    inRange(hsv, lower_blue, upper_blue, mask_roi);
//
//    // 形态学操作（只在 ROI 区域内执行，省时）
//    Mat kernel = getStructuringElement(MORPH_RECT, Size(9, 9));
//    morphologyEx(mask_roi, mask_roi, MORPH_OPEN, kernel);
//    morphologyEx(mask_roi, mask_roi, MORPH_CLOSE, kernel);//等会儿注释了看看效果
//
//    // 查找轮廓
//    vector<vector<Point>> contours;
//    findContours(mask_roi, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
//
//    vector<Point> raw_centers;
//    for (const auto& cnt : contours) {
//        double area = contourArea(cnt);
//        if (area > 40.0) { // 阈值略微调小，因为部分被遮挡的箭头面积变小了
//            Moments m = moments(cnt);
//            if (m.m00 != 0) {
//                int cx = static_cast<int>(m.m10 / m.m00);
//                // 【注意】计算出的 Y 坐标是相对于 ROI 区域的，必须把切掉的 offset 加回来，还原到原图坐标
//                int cy = static_cast<int>(m.m01 / m.m00) + roi_offset_y;
//                raw_centers.push_back(Point(cx, cy));
//            }
//        }
//    }
//
//    // 按 Y 坐标降序排序（离车最近的点在前面）
//    sort(raw_centers.begin(), raw_centers.end(), [](const Point& a, const Point& b) {
//        return a.y > b.y;
//        });
//
//
//    vector<Point> filtered_centers;
//    if (!raw_centers.empty()) {
//        filtered_centers.push_back(raw_centers[0]); // 假设离车最近的第一个点是可靠的
//
//        for (size_t i = 1; i < raw_centers.size(); ++i) {
//            Point prev = filtered_centers.back();
//            Point curr = raw_centers[i];
//
//            // A. 防分裂处理：如果两个点 Y 坐标几乎一样（箭头被行人腿从中踩断成两半）
//            if (abs(curr.y - prev.y) < 8) {
//                continue; // 直接忽略这个多余的碎片，防止轨迹扭曲
//            }
//
//            // B. 防跳变处理：计算横向漂移斜率
//            float dx = abs(curr.x - prev.x);
//            float dy = abs(curr.y - prev.y);
//
//            // 如果 Y 前进了很少，X 却发生了巨大的突变（斜率异常），说明这是被遮挡导致的偏离点或噪点
//            if (dx > src.cols / 3 || dy > src.rows / 3) {
//                // cout << "剔除离群点: (" << curr.x << "," << curr.y << ")" << endl;
//                continue;
//            }
//
//            filtered_centers.push_back(curr);
//        }
//    }
//
//    return filtered_centers;
//}



//vector<Point> LineTracker::FitTrajectory_Bezier(const vector<Point>& pts, Mat& frame) {
//
//    vector<Point> curve_pts;
//    if (pts.size() < 3) {
//
//        return curve_pts; // 经过过滤后如果点数太少，就不强行拟合了
//    }
//
//    int n = pts.size();
//    Mat A(n, 3, CV_32F);
//    Mat B(n, 1, CV_32F);
//
//    for (int i = 0; i < n; ++i) {
//        float y = static_cast<float>(pts[i].y);
//        A.at<float>(i, 0) = y * y;
//        A.at<float>(i, 1) = y;
//        A.at<float>(i, 2) = 1.0f;
//        B.at<float>(i, 0) = static_cast<float>(pts[i].x);
//    }
//
//    Mat coeffs;
//    solve(A, B, coeffs, DECOMP_SVD);
//
//    float a = coeffs.at<float>(0, 0);
//    float b = coeffs.at<float>(1, 0);
//    float c = coeffs.at<float>(2, 0);
//
//    int y_start = pts.front().y;
//    int y_end = pts.back().y;
//
//    if (y_start <= y_end) return curve_pts;
//
//    for (int y = y_start; y >= y_end; y -= 5) {
//        int x = static_cast<int>(round(a * y * y + b * y + c));
//        if (x >= 0 && x < frame.cols) {
//            curve_pts.push_back(Point(x, y));
//        }
//    }
//    return curve_pts;
//}