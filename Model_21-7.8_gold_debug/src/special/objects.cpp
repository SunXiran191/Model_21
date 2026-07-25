#include "objects.hpp"
#include "../imgprocess/TrajectoryFitter.hpp"
#include <algorithm>
#include <cmath>

using namespace std;
using namespace cv;

Objects::Objects() {}

Objects::Objects(Config &config)
{
    this->gold_dist_forward = config.GOLD_DIST_FORWARD;
    this->gold_dist_backward = config.GOLD_DIST_BACKWARD;
    this->car_dist_forward = config.CAR_DIST_FORWARD;
    this->car_dist_backward = config.CAR_DIST_BACKWARD;
    this->car_avoid_dist = config.car_avoid_dist;
    this->max_gold_track_dist = config.max_gold_track_dist;
    this->confidence_threshold = config.confidence_threshold;
}

// ============================================================================
// 车辆避障点计算（固定距离 + 方向迟滞）
// ============================================================================
cv::Point Objects::computeCarAvoidPoint(
    const cv::Point &car_center,
    const vector<cv::Point> &trajectory,
    int trajectory_size)
{
    cv::Point target_point;

    // 找与障碍物 Y 坐标最接近的轨迹点
    int closest_idx = 0;
    int min_y_diff = 100000;
    for (int i = 0; i < trajectory_size; ++i)
    {
        int y_diff = abs(trajectory[i].y - car_center.y);
        if (y_diff < min_y_diff)
        {
            min_y_diff = y_diff;
            closest_idx = i;
        }
    }

    int trajectory_ref_x = trajectory[closest_idx].x;
    int offset_x = car_center.x - trajectory_ref_x; // 正=障碍在赛道右侧, 负=左侧
    int avoid_dist = car_avoid_dist;

    // ── 迟滞方向决策（防左右跳变） ──
    int new_dir = (offset_x < 0) ? 1 : -1; // 1=右避（障碍在左）, -1=左避（障碍在右）

    if (avoid_dir_last_ == 0)
    {
        avoid_dir_last_ = new_dir;
        avoid_dir_lock_cnt_ = 1;
    }
    else if (abs(offset_x) < avoid_dir_dead_zone)
    {
        avoid_dir_lock_cnt_++;
    }
    else if (new_dir != avoid_dir_last_)
    {
        avoid_dir_lock_cnt_++;
        if (avoid_dir_lock_cnt_ >= avoid_dir_flip_thresh)
        {
            avoid_dir_last_ = new_dir;
            avoid_dir_lock_cnt_ = 0;
            avoid_x_ema_ = -1.0f;
        }
    }
    else
    {
        avoid_dir_lock_cnt_ = 0;
    }

    int avoid_dir = avoid_dir_last_;

    // ── 计算避障点 ──
    if (avoid_dir == 0 || avoid_dist == 0)
    {
        avoid_x_ema_ = -1.0f;
        target_point.x = clip(trajectory_ref_x, 0, IMAGE_W - 1);
        return cv::Point(target_point.x, car_center.y);
    }

    float raw_x = (float)trajectory_ref_x + avoid_dir * avoid_dist;

    // ── EMA 平滑（减少帧间抖动） ──
    if (avoid_x_ema_ < 0.0f)
    {
        avoid_x_ema_ = raw_x;
    }
    else
    {
        avoid_x_ema_ = avoid_ema_alpha_ * raw_x + (1.0f - avoid_ema_alpha_) * avoid_x_ema_;
    }

    target_point.x = clip((int)avoid_x_ema_, 0, IMAGE_W - 1);
    return cv::Point(target_point.x, car_center.y);
}

void Objects::processAllObjects(
    cv::Mat &src_img,
    const vector<cv::Point> &t_gold_point,
    const vector<cv::Point> &t_car_point,
    vector<cv::Point> &trajectory,
    int &trajectory_size)
{
    if (trajectory_size < 3)
        return;

    // 重置金币夹角标志（每帧重新计算）
    gold_angles.clear();
    gold_angle_flag = false;
    gold_seq_y_min = 0.0f;
    gold_seq_y_max = 0.0f;

    targets.clear();
    float max_y = static_cast<float>(IMAGE_H - 1);

    vector<ObjectsTargetInfo> car_targets;

    // ================== 优先处理障碍物 ==================

    // ── 车辆避障 ──
    avoid_dir_last_ = 0;
    avoid_dir_lock_cnt_ = 0;
    avoid_x_ema_ = -1.0f;
    for (const auto &pt : t_car_point)
    {
        cv::Point offset = computeCarAvoidPoint(pt, trajectory, trajectory_size);
        ObjectsTargetInfo t;
        t.pt = offset;
        t.y_low = std::max(0.f, static_cast<float>(pt.y - car_dist_forward));
        t.y_high = std::min(max_y, static_cast<float>(pt.y + car_dist_backward));
        car_targets.push_back(t);
    }

    // ================== 安全校验后处理金币 ==================

    float coin_safe_buffer = 10.0f;         // 金币安全缓冲距离
    vector<ObjectsTargetInfo> gold_targets; // 用一个单独的数组存金币

    for (const auto &pt : t_gold_point)
    {
        // ── 0. 距离赛道过远过滤：找轨迹中最接近该金币Y的点，检查X偏差 ──
        bool near_track = true;
        if (max_gold_track_dist > 0 && trajectory_size >= 2)
        {
            int best_i = 0, min_dy = 100000;
            for (int i = 0; i < trajectory_size; ++i)
            {
                int dy = abs(trajectory[i].y - pt.y);
                if (dy < min_dy)
                {
                    min_dy = dy;
                    best_i = i;
                }
            }
            int track_x_at_gold = trajectory[best_i].x;
            if (abs(pt.x - track_x_at_gold) > max_gold_track_dist)
            {
                near_track = false;
            }
        }
        if (!near_track)
            continue;

        // ── 1. 安全检查：金币是否与车辆危险区重叠 ──
        bool is_safe = true;
        for (const auto &ct : car_targets)
        {
            if (pt.y >= (ct.y_low - coin_safe_buffer) && pt.y <= (ct.y_high + coin_safe_buffer))
            {
                is_safe = false;
                break;
            }
        }

        if (is_safe)
        {
            ObjectsTargetInfo t;
            t.pt = pt;
            t.y_low = std::max(0.f, static_cast<float>(pt.y - this->gold_dist_forward));
            t.y_high = std::min(max_y, static_cast<float>(pt.y + this->gold_dist_backward));
            gold_targets.push_back(t);
        }
    }

    // ── 金币夹角序列计算（预判转弯） ──
    computeGoldAngleSeq(gold_targets);

    if (gold_targets.empty() && car_targets.empty())
    {
        return;
    }

    // ================== 拟合目标曲线 ==================
    vector<cv::Point> merged;

    // 保留不在屏蔽Y范围内的原生巡线点
    for (const auto &pt : trajectory)
    {
        bool masked = false;
        // 检查是否被金币区屏蔽
        for (const auto &t : gold_targets)
        {
            if (pt.y >= t.y_low && pt.y <= t.y_high)
            {
                masked = true;
                break;
            }
        }
        // 检查是否被车辆危险区屏蔽
        for (const auto &t : car_targets)
        {
            if (pt.y >= t.y_low && pt.y <= t.y_high)
            {
                masked = true;
                break;
            }
        }

        if (!masked)
            merged.push_back(pt);
    }

    // 使用密集点墙防止被橡胶带平滑抹平
    int gold_wall_length = 14; // 墙的半高度（像素）
    int gold_wall_step = 2;    // 每隔多少像素放一个点
    for (const auto &t : gold_targets)
    {
        for (int dy = -gold_wall_length; dy <= gold_wall_length; dy += gold_wall_step)
        {
            int wy = clip(t.pt.y + dy, 0, static_cast<int>(max_y));
            merged.push_back(cv::Point(t.pt.x, wy));
        }
    }

    // ── 车辆避障密集点墙 ──
    int car_wall_length = 44; // 墙的半高度（像素）
    int car_wall_step = 4;    // 每隔多少像素放一个点
    for (const auto &ct : car_targets)
    {
        for (int dy = -car_wall_length; dy <= car_wall_length; dy += car_wall_step)
        {
            int wy = clip(ct.pt.y + dy, 0, static_cast<int>(max_y));
            merged.push_back(cv::Point(ct.pt.x, wy));
        }
    }

    sort(merged.begin(), merged.end(),
         [](const cv::Point &a, const cv::Point &b)
         { return a.y > b.y; });

    // 应用"橡胶带"平滑
    vector<cv::Point> rubber_band_curve = generateRubberBandCurve(merged, 10, 5);

    // 等距采样，恢复点距
    trajectory = resampleEquidistant(rubber_band_curve, 10.0f);

    trajectory_size = trajectory.size();
}

// 橡胶带算法：线性加密+多次滑动平均滤波
std::vector<cv::Point> Objects::generateRubberBandCurve(const std::vector<cv::Point> &input_points, int window_size, int iterations)
{
    // if (input_points.size() < 3)
    //     return input_points;

    // 线性加密
    std::vector<cv::Point2f> dense_pts;
    float step = 3.0f;
    for (size_t i = 0; i < input_points.size() - 1; ++i)
    {
        cv::Point2f p1(input_points[i].x, input_points[i].y);
        cv::Point2f p2(input_points[i + 1].x, input_points[i + 1].y);
        float dist = std::hypot(p2.x - p1.x, p2.y - p1.y);
        int num_steps = std::max(1, (int)(dist / step));
        for (int j = 0; j < num_steps; ++j)
        {
            dense_pts.push_back(p1 + (p2 - p1) * ((float)j / num_steps));
        }
    }
    dense_pts.push_back(cv::Point2f(input_points.back().x, input_points.back().y));

    // 多次滑动平均滤波
    std::vector<cv::Point2f> smoothed = dense_pts;
    std::vector<cv::Point2f> temp = dense_pts;
    for (int iter = 0; iter < iterations; ++iter)
    {
        for (size_t i = 1; i < smoothed.size() - 1; ++i)
        {
            int half_win = window_size / 2;
            int start = std::max(0, (int)i - half_win);
            int end = std::min((int)smoothed.size() - 1, (int)i + half_win);

            float sum_x = 0, sum_y = 0;
            for (int w = start; w <= end; ++w)
            {
                sum_x += smoothed[w].x;
                sum_y += smoothed[w].y;
            }
            int count = end - start + 1;
            temp[i] = cv::Point2f(sum_x / count, sum_y / count);
        }
        temp[0] = dense_pts[0];
        temp.back() = dense_pts.back();
        smoothed = temp;
    }

    std::vector<cv::Point> output;
    for (const auto &p : smoothed)
    {
        output.push_back(cv::Point(cvRound(p.x), cvRound(p.y)));
    }
    return output;
}

// 等距采样
std::vector<cv::Point> Objects::resampleEquidistant(const std::vector<cv::Point> &input, float step_dist)
{
    if (input.size() < 2)
        return input;
    std::vector<cv::Point> output;
    output.push_back(input[0]);
    float current_dist = 0.0f;

    for (size_t i = 1; i < input.size(); ++i)
    {
        float dx = input[i].x - input[i - 1].x;
        float dy = input[i].y - input[i - 1].y;
        float segment_len = std::hypot(dx, dy);

        while (current_dist + step_dist <= segment_len)
        {
            float ratio = (current_dist + step_dist) / segment_len;
            float nx = input[i - 1].x + dx * ratio;
            float ny = input[i - 1].y + dy * ratio;
            output.push_back(cv::Point(cvRound(nx), cvRound(ny)));
            current_dist += step_dist;
        }
        current_dist -= segment_len;
    }
    output.push_back(input.back());
    return output;
}

// ============================================================================
// 金币夹角序列计算
// 序列顺序：金币屏蔽区前端点 → 金币 → 金币屏蔽区后端点（按y从大到小，由近到远）
// 计算序列相邻点构成的线段之间的内角
// ============================================================================
void Objects::computeGoldAngleSeq(const std::vector<ObjectsTargetInfo> &gold_targets)
{
    if (gold_targets.empty())
        return;

    int n = static_cast<int>(gold_targets.size());

    // ── 检测不同金币之间 Y 范围是否重叠 ──
    // 重叠的硬币可能位于分叉路径上，其 fp/bp 不计入序列，避免干扰角度计算
    std::vector<bool> skip_fp(n, false);
    std::vector<bool> skip_bp(n, false);

    for (int i = 0; i < n; ++i)
    {
        float y_lo_i = gold_targets[i].y_low;
        float y_hi_i = gold_targets[i].y_high;
        for (int j = i + 1; j < n; ++j)
        {
            float y_lo_j = gold_targets[j].y_low;
            float y_hi_j = gold_targets[j].y_high;
            // 区间重叠判定：max(lo1, lo2) <= min(hi1, hi2)
            if (std::max(y_lo_i, y_lo_j) <= std::min(y_hi_i, y_hi_j))
            {
                skip_fp[i] = true;
                skip_bp[i] = true;
                skip_fp[j] = true;
                skip_bp[j] = true;
            }
        }
    }

    // ── 构建序列：每个金币贡献最多3个点（前端、中心、后端），按Y从大到小（近→远） ──
    struct GoldPoint
    {
        cv::Point pt;
        float y_sort; // 排序用
    };
    std::vector<GoldPoint> goldpoint;
    goldpoint.reserve(n * 3);

    for (int i = 0; i < n; ++i)
    {
        const ObjectsTargetInfo &gt = gold_targets[i];

        // 前端点（y_high = 更靠近车 = Y更大）—— 与其它金币Y范围重叠时跳过
        if (!skip_fp[i])
        {
            GoldPoint fp;
            fp.pt = cv::Point(gt.pt.x, static_cast<int>(gt.y_high));
            fp.y_sort = gt.y_high;
            goldpoint.push_back(fp);
        }

        // 金币中心点（始终保留）
        GoldPoint cp;
        cp.pt = gt.pt;
        cp.y_sort = static_cast<float>(gt.pt.y);
        goldpoint.push_back(cp);

        // 后端点（y_low = 更远 = Y更小）—— 与其它金币Y范围重叠时跳过
        if (!skip_bp[i])
        {
            GoldPoint bp;
            bp.pt = cv::Point(gt.pt.x, static_cast<int>(gt.y_low));
            bp.y_sort = gt.y_low;
            goldpoint.push_back(bp);
        }
    }

    // 按Y从大到小（近→远）排序
    std::sort(goldpoint.begin(), goldpoint.end(),
              [](const GoldPoint &a, const GoldPoint &b)
              { return a.y_sort > b.y_sort; });

    // 提取排序后的点序列
    std::vector<cv::Point> seq;
    seq.reserve(goldpoint.size());
    for (size_t i = 0; i < goldpoint.size(); ++i)
        seq.push_back(goldpoint[i].pt);

    // ── 记录序列的Y范围 ──
    gold_seq_y_min = static_cast<float>(seq.back().y);
    gold_seq_y_max = static_cast<float>(seq.front().y);

    // ── 计算相邻线段的内角（至少需要3个点） ──
    if (seq.size() < 3)
    {
        return;
    }

    // 对每个中间点计算内角，全部存入数组
    for (size_t i = 1; i < seq.size() - 1; ++i)
    {
        const cv::Point &A = seq[i - 1];
        const cv::Point &B = seq[i];
        const cv::Point &C = seq[i + 1];

        // 向量 BA 和 BC
        float BAx = static_cast<float>(A.x - B.x);
        float BAy = static_cast<float>(A.y - B.y);
        float BCx = static_cast<float>(C.x - B.x);
        float BCy = static_cast<float>(C.y - B.y);

        float lenBA = std::sqrt(BAx * BAx + BAy * BAy);
        float lenBC = std::sqrt(BCx * BCx + BCy * BCy);

        if (lenBA < 1e-6f || lenBC < 1e-6f)
            continue;

        // 内角 cos = (BA·BC) / (|BA| * |BC|)
        float dot = BAx * BCx + BAy * BCy;
        float cos_angle = dot / (lenBA * lenBC);
        cos_angle = std::max(-1.0f, std::min(1.0f, cos_angle));
        float angle_rad = std::acos(cos_angle);
        float angle_deg = angle_rad * 180.0f / PI;

        gold_angles.push_back({angle_deg, static_cast<float>(C.y), static_cast<float>(A.y)});
    }
}
