#include "go_stop.hpp"
#include "../imgprocess/transform.hpp"
#include "../special/objects.hpp"
#include <algorithm>
#include <cmath>

using namespace std;
using namespace cv;

GoStop::GoStop() {}

GoStop::GoStop(Config &config)
{
    this->go_stop_dist_forward = config.go_stop_dist_forward;
    this->go_stop_dist_backward = config.go_stop_dist_backward;
    this->confidence_threshold = config.confidence_threshold;
}

// ============================================================================
// 将 STOP/GO 检测框填充到语义分割掩码中（设为道路区域 255）
// ============================================================================
void GoStop::paintBoxesToSegMask(
    cv::Mat &segmask,
    const std::vector<PredictResult> &stop_results,
    const std::vector<PredictResult> &go_results)
{
    if (segmask.empty())
        return;

    auto paint = [&](const std::vector<PredictResult> &results)
    {
        for (const auto &r : results)
        {
            int x1 = clip(r.x1, 0, segmask.cols - 1);
            int y1 = clip(r.y1, 0, segmask.rows - 1);
            int x2 = clip(r.x2, 0, segmask.cols - 1);
            int y2 = clip(r.y2, 0, segmask.rows - 1);
            if (x2 <= x1 || y2 <= y1)
                continue;
            // 将检测框区域填充为道路（255）
            cv::Rect roi(x1, y1, x2 - x1, y2 - y1);
            segmask(roi).setTo(255);
        }
    };

    paint(stop_results);
    paint(go_results);
}

// ============================================================================
// 对 STOP/GO 标志中心点做直线拟合，返回归一化方向向量（指向图像顶部）
// ============================================================================
cv::Point2f GoStop::fitLineThroughCenters(const std::vector<cv::Point> &centers) const
{
    if (centers.size() < 2)
    {
        // 单点或无点：使用竖直方向
        return cv::Point2f(0.0f, -1.0f);
    }

    // 最小二乘拟合 x = k*y + b（y 方向变化大，以此作自变量更稳定）
    int n = static_cast<int>(centers.size());
    double sum_y = 0.0, sum_x = 0.0, sum_yy = 0.0, sum_xy = 0.0;
    for (int i = 0; i < n; i++)
    {
        double xi = static_cast<double>(centers[i].x);
        double yi = static_cast<double>(centers[i].y);
        sum_y += yi;
        sum_x += xi;
        sum_yy += yi * yi;
        sum_xy += xi * yi;
    }

    double denom = n * sum_yy - sum_y * sum_y;
    if (std::abs(denom) < 1e-6)
        return cv::Point2f(0.0f, -1.0f);

    double k = (n * sum_xy - sum_x * sum_y) / denom;

    // 方向向量沿 y 减小（图像上方）：dy=-1, dx=-k
    cv::Point2f dir(-static_cast<float>(k), -1.0f);
    float len = std::sqrt(dir.x * dir.x + dir.y * dir.y);
    if (len > 1e-6f)
    {
        dir.x /= len;
        dir.y /= len;
    }
    return dir;
}

// ============================================================================
// 主入口：抹去标志遮挡区的轨迹点，用 IPM 中心拟合线替代
// ============================================================================
void GoStop::processGoStop(
    const std::vector<cv::Point> &stop_point,
    const std::vector<cv::Point> &go_point,
    std::vector<cv::Point> &trajectory,
    int &trajectory_size)
{
    if (trajectory_size < 3)
        return;

    // 收集所有 STOP + GO 的 IPM 中心坐标
    std::vector<cv::Point> all_centers;
    all_centers.reserve(stop_point.size() + go_point.size());
    for (const auto &pt : stop_point)
        all_centers.push_back(pt);
    for (const auto &pt : go_point)
        all_centers.push_back(pt);

    if (all_centers.empty())
        return;

    // 按 y 降序排列（底部→顶部，由近到远）
    std::sort(all_centers.begin(), all_centers.end(),
              [](const cv::Point &a, const cv::Point &b)
              { return a.y > b.y; });

    // 屏蔽区 y 范围：最下方中心向下延伸 backward，最上方中心向上延伸 forward
    int bottom_y = all_centers.front().y;
    int top_y = all_centers.back().y;
    float max_y_f = static_cast<float>(IMAGE_H - 1);
    float y_low = std::max(0.0f, static_cast<float>(bottom_y + go_stop_dist_backward));
    float y_high = std::min(max_y_f, static_cast<float>(top_y - go_stop_dist_forward));

    if (y_high >= y_low)
        return;

    // 拟合直线方向
    cv::Point2f slope_dir = fitLineThroughCenters(all_centers);

    // 沿拟合直线生成替换点（从 y_high 到 y_low，步长 4 像素）
    std::vector<cv::Point> fitted_pts;
    int fit_steps = static_cast<int>((y_low - y_high) / 4.0f);
    if (fit_steps < 2)
        fit_steps = 2;

    // 取所有中心的平均 x 作为锚点，沿斜率方向生成点
    float avg_x = 0.0f, avg_y = 0.0f;
    for (const auto &c : all_centers)
    {
        avg_x += c.x;
        avg_y += c.y;
    }
    avg_x /= all_centers.size();
    avg_y /= all_centers.size();

    for (int i = 0; i <= fit_steps; i++)
    {
        float t = y_high + static_cast<float>(i) * (y_low - y_high) / fit_steps;
        // 从平均中心沿斜率方向推算 x：斜率 dir = (dx, dy)，沿 y 方向投影
        float dy = t - avg_y;
        float dx = (slope_dir.x / slope_dir.y) * dy; // slope_dir.y 指向 -1，正常
        int px = clip(static_cast<int>(avg_x + dx), 0, IMAGE_W - 1);
        int py = clip(static_cast<int>(t), 0, static_cast<int>(max_y_f));
        fitted_pts.push_back(cv::Point(px, py));
    }

    // ================== 合并轨迹 ==================
    std::vector<cv::Point> merged;

    // 保留屏蔽区外的原生轨迹点
    for (const auto &pt : trajectory)
    {
        if (pt.y < y_high || pt.y > y_low)
            merged.push_back(pt);
    }

    // 插入标志中心 + 拟合线点
    for (const auto &pt : all_centers)
        merged.push_back(pt);
    for (const auto &pt : fitted_pts)
        merged.push_back(pt);

    // 按 y 降序排列
    std::sort(merged.begin(), merged.end(),
              [](const cv::Point &a, const cv::Point &b)
              { return a.y > b.y; });

    if (merged.size() < 3)
        return;

    // 橡胶带平滑 + 等距采样
    std::vector<cv::Point> rubber_band_curve = Objects::generateRubberBandCurve(merged, 10, 5);
    trajectory = Objects::resampleEquidistant(rubber_band_curve, 10.0f);
    trajectory_size = static_cast<int>(trajectory.size());
}
