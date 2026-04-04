#include "TrajectoryFitter.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>

// 简单的多项式拟合实现（最小二乘法）
vector<Point> LineTracker::FitTrajectory_LSM(const vector<Point> &pts, Mat &frame)
{
    vector<Point> curve_pts;
    if (pts.size() < 3)
        return curve_pts; // 经过过滤后如果点数太少，就不强行拟合了

    int n = pts.size();
    Mat A(n, 3, CV_32F);
    Mat B(n, 1, CV_32F);

    for (int i = 0; i < n; ++i)
    {
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

    if (y_start <= y_end)
        return curve_pts;

    for (int y = y_start; y >= y_end; y -= 5)
    {
        int x = static_cast<int>(round(a * y * y + b * y + c));
        if (x >= 0 && x < frame.cols)
        {
            curve_pts.push_back(Point(x, y));
        }
    }
    return curve_pts;
}

// lowess局部加权回归拟合，适合处理噪点较多的情况，计算量较大
vector<Point> LineTracker::FitTrajectory_LOWESS(int point_count, const vector<Point> &pts, Mat &frame)
{
    vector<Point> curve_pts;
    int n = pts.size();
    if (n < 2)
        return pts;

    // 平滑窗口大小 (占总点数的比例)
    double span = 0.4;
    int window_size = max(3, (int)(n * span));

    for (int i = 0; i < point_count; ++i)
    {
        double t = (double)i / (point_count - 1) * (n - 1); // 目标插值点位置

        double sum_w = 0, sum_wt = 0, sum_wt2 = 0, sum_wx = 0, sum_wy = 0, sum_wtx = 0, sum_wty = 0;

        for (int j = 0; j < n; ++j)
        {
            double diff = abs(t - j);
            if (diff < window_size)
            {
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
        if (abs(det) > 1e-6)
        {
            double x_fit = (sum_wt2 * sum_wx - sum_wt * sum_wtx) / det +
                           (sum_w * sum_wtx - sum_wt * sum_wx) / det * t;
            double y_fit = (sum_wt2 * sum_wy - sum_wt * sum_wty) / det +
                           (sum_w * sum_wty - sum_wt * sum_wy) / det * t;
            curve_pts.push_back(Point(cvRound(x_fit), cvRound(y_fit)));
        }
    }
    return curve_pts;
}

// 多项式拟合，计算量小但对噪点敏感
vector<Point> LineTracker::FitTrajectory_Poly(int point_count, const vector<Point> &pts, Mat &frame)
{
    vector<Point> curve_pts;
    int n = pts.size();
    int degree = 3; // 多项式阶数
    if (n <= degree)
        return pts;

    auto solvePoly = [&](const vector<double> &vals)
    {
        Mat A(n, degree + 1, CV_64F);
        Mat B(n, 1, CV_64F);
        for (int i = 0; i < n; i++)
        {
            double t = (double)i / (n - 1);
            for (int j = 0; j <= degree; j++)
            {
                A.at<double>(i, j) = pow(t, j);
            }
            B.at<double>(i, 0) = vals[i];
        }
        Mat coeffs;
        solve(A, B, coeffs, DECOMP_SVD);
        return coeffs;
    };

    vector<double> px, py;
    for (const auto &p : pts)
    {
        px.push_back(p.x);
        py.push_back(p.y);
    }

    Mat coeffX = solvePoly(px);
    Mat coeffY = solvePoly(py);

    for (int i = 0; i < point_count; i++)
    {
        double t = (double)i / (point_count - 1);
        double rx = 0, ry = 0;
        for (int j = 0; j <= degree; j++)
        {
            rx += coeffX.at<double>(j, 0) * pow(t, j);
            ry += coeffY.at<double>(j, 0) * pow(t, j);
        }
        curve_pts.push_back(Point(cvRound(rx), cvRound(ry)));
    }
    return curve_pts;
}

// 高斯过程回归，适合处理复杂曲线，计算量较大
vector<Point> LineTracker::FitTrajectory_GPR(int point_count, const vector<Point> &pts, Mat &frame)
{
    vector<Point> curve_pts;
    int n = pts.size();
    if (n < 2)
        return pts;

    // GPR 超参数
    double l = 0.6;       // 长度尺度 (Length scale)
    double sigma_f = 1.0; // 信号方差
    double sigma_n = 0.1; // 噪声标准差

    auto kernel = [&](double t1, double t2)
    {
        return sigma_f * exp(-pow(t1 - t2, 2) / (2 * l * l));
    };

    // 构建协方差矩阵 K
    Mat K(n, n, CV_64F);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            double t_i = (double)i / (n - 1);
            double t_j = (double)j / (n - 1);
            K.at<double>(i, j) = kernel(t_i, t_j) + (i == j ? sigma_n * sigma_n : 0);
        }
    }

    Mat K_inv = K.inv(DECOMP_CHOLESKY);

    Mat Y_x(n, 1, CV_64F), Y_y(n, 1, CV_64F);
    for (int i = 0; i < n; i++)
    {
        Y_x.at<double>(i, 0) = pts[i].x;
        Y_y.at<double>(i, 0) = pts[i].y;
    }

    // 预测
    for (int i = 0; i < point_count; i++)
    {
        double t_star = (double)i / (point_count - 1);
        Mat k_star(1, n, CV_64F);
        for (int j = 0; j < n; j++)
        {
            k_star.at<double>(0, j) = kernel(t_star, (double)j / (n - 1));
        }

        Mat mu_x = k_star * K_inv * Y_x;
        Mat mu_y = k_star * K_inv * Y_y;

        curve_pts.push_back(Point(cvRound(mu_x.at<double>(0, 0)),
                                  cvRound(mu_y.at<double>(0, 0))));
    }
    return curve_pts;
}

// 分段贝塞尔曲线
//  感觉不能用，很难排除噪点/极端点
vector<Point> LineTracker::FitTrajectory_Bezier(int point_count, const vector<Point> &lane_points)
{
    vector<Point> output;
    // 点数不足时直接返回原始点
    if (point_count < 2 || lane_points.empty() || lane_points.size() != point_count)
    {
        return lane_points;
    }

    // 平滑步长（控制拟合曲线的点数，值越小曲线越平滑）
    const double dt = 0.02; // 建议值：0.01~0.05，0.02兼顾平滑度和效率

    // 不同点数的适配逻辑
    // 1 只有2个点：线性拟合
    if (point_count == 2)
    {
        for (double t = 0; t <= 1 + 1e-6; t += dt)
        {
            int x = round((1 - t) * lane_points[0].x + t * lane_points[1].x);
            int y = round((1 - t) * lane_points[0].y + t * lane_points[1].y);
            output.push_back(Point(x, y));
        }
        return output;
    }

    // 2 只有3个点：二次贝塞尔拟合
    if (point_count == 3)
    {
        for (double t = 0; t <= 1 + 1e-6; t += dt)
        {
            double t1 = 1 - t;
            int x = round(t1 * t1 * lane_points[0].x + 2 * t1 * t * lane_points[1].x + t * t * lane_points[2].x);
            int y = round(t1 * t1 * lane_points[0].y + 2 * t1 * t * lane_points[1].y + t * t * lane_points[2].y);
            output.push_back(Point(x, y));
        }
        return output;
    }

    // 3 点数≥4：分段三次贝塞尔拟合
    // 辅助函数：计算单段三次贝塞尔点
    auto calcCubicBezier = [dt](const vector<Point> &ctrl_pts) -> vector<Point>
    {
        vector<Point> segment;
        for (double t = 0; t <= 1 + 1e-6; t += dt)
        {
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
    for (size_t i = 0; i <= lane_points.size() - 4; i += 3)
    {
        // 提取当前段的4个控制点
        vector<Point> ctrl_pts(lane_points.begin() + i, lane_points.begin() + i + 4);
        vector<Point> segment_pts = calcCubicBezier(ctrl_pts);
        output.insert(output.end(), segment_pts.begin(), segment_pts.end());
    }

    // 处理最后不足4个点的剩余部分
    if (lane_points.size() % 3 != 1 && lane_points.size() > 4)
    {
        vector<Point> last_ctrl_pts(lane_points.end() - 4, lane_points.end());
        vector<Point> last_segment = calcCubicBezier(last_ctrl_pts);
        output.insert(output.end(), last_segment.begin(), last_segment.end());
    }

    return output;
}