#include "branch.hpp"
#include "../common/utils.hpp"
#include "../../res/configs/param.hpp"
#include "../thread/thread.hpp" // g_ocr_trigger, g_ocr_nav_result, OcrNavResult
#include <algorithm>
#include <atomic>
#include <cmath>

using namespace std;
using namespace cv;

Branch::Branch()
    : has_branch(false),
      is_t_junction(false),
      multi_contour_count(0),
      confidence_threshold(0.44f),
      branch_min_pixels(8), 
      branch_min_rows(8),
      t_junction_margin(20.0f),
      t_junction_row_threshold(8),
      current_state_(BRANCH_NONE),
      state_frame_count_(0),
      sign_detect_count_(0),
      sign_lost_count_(0),
      branch_in_count_(0),
      no_fork_count_(0),
      rectangular_count_(0),
      initial_sign_area_(0.0f),
      initial_area_set_(false)
{
}

void Branch::init(const Config &config)
{
    confidence_threshold = config.confidence_threshold;
    branch_detect_frame_threshold = config.branch_detect_frame_threshold;
    branch_in_frame_threshold = config.branch_in_frame_threshold;
    branch_out_frame_threshold = config.branch_out_frame_threshold;
    branch_over_frame_threshold = config.branch_over_frame_threshold;
    branch_none_frame_threshold = config.branch_none_frame_threshold;
    t_junction_margin = config.t_junction_margin;
    t_junction_row_threshold = config.t_junction_row_threshold;
    branch_ocr_speed = config.branch_ocr_speed;
    branch_sign_width_threshold = config.branch_sign_width_threshold;
    branch_sign_edge_x_margin = config.branch_sign_edge_x_margin;
    branch_sign_edge_y_margin = config.branch_sign_edge_y_margin;
    branch_sign_area_ratio_x2 = config.branch_sign_area_ratio_x2;
}

void Branch::reset_frame()
{
    has_branch = false;
    is_t_junction = false;
    multi_contour_count = 0;
    branch_rows_.clear();
}

void Branch::detect_t_junction(const std::vector<std::pair<int, int>> &row_edges, int total_rows)
{
    is_t_junction = false;

    if (total_rows <= 0)
        return;

    // 底边 1/5 区域的平均左右边缘坐标作为基线
    int baseline_rows = std::max(1, total_rows / 5);
    int sum_left = 0, sum_right = 0;
    int valid_baseline = 0;
    for (int i = 0; i < baseline_rows; ++i)
    {
        int lx = row_edges[i].first;
        int rx = row_edges[i].second;
        if (lx >= 0 && rx >= 0)
        {
            sum_left += lx;
            sum_right += rx;
            valid_baseline++;
        }
    }

    if (valid_baseline == 0)
        return;

    float avg_left = (float)sum_left / valid_baseline;
    float avg_right = (float)sum_right / valid_baseline;

    int exceed_count = 0;
    for (int i = baseline_rows; i < total_rows; ++i)
    {
        int lx = row_edges[i].first;
        int rx = row_edges[i].second;
        if (lx >= 0 && rx >= 0)
        {
            // 左边缘比基线更左，或右边缘比基线更右 → 道路变宽
            if (lx < avg_left - t_junction_margin && rx > avg_right + t_junction_margin)
                exceed_count++;
        }
    }

    // 超出行数达到阈值
    is_t_junction = (exceed_count >= t_junction_row_threshold);
}

void Branch::check_branch(const std::vector<PredictResult> &predict_result)
{
    // 仅在 BRANCH_NONE 状态下响应岔路标志
    if (current_state_ != BRANCH_NONE)
        return;

    // 检测本帧是否识别到岔路标志
    bool detected_this_frame = false;
    int bx1 = 0, by1 = 0, bx2 = 0, by2 = 0;
    for (const auto &obj : predict_result)
    {
        if (obj.class_id == CLASS_ID_BRANCHSIGN && obj.score >= confidence_threshold)
        {
            detected_this_frame = true;
            bx1 = obj.x1;
            by1 = obj.y1;
            bx2 = obj.x2;
            by2 = obj.y2;
            break;
        }
    }

    if (detected_this_frame)
    {
        // 检查检测框各边是否都距画面边缘足够远，且宽度大于阈值
        bool edges_clear = (bx2 - bx1) > branch_sign_width_threshold;

        if (edges_clear)
        {
            sign_detect_count_++;
            std::cout << "[Branch] edges_clear count=" << sign_detect_count_ << std::endl;

            if (sign_detect_count_ >= branch_detect_frame_threshold)
            {
                current_state_ = BRANCH_DETECTED;
                state_frame_count_ = 0;
                sign_detect_count_ = 0;
                sign_lost_count_ = 0;
                branch_in_count_ = 0;
                no_fork_count_ = 0;
                rectangular_count_ = 0;
                detected_timeout_cnt = 0;
                // 重置状态，等 DETECTED 中首次满足 edges_clear 再记录初始面积
                initial_area_set_ = false;
                initial_sign_area_ = 0.0f;
                g_ocr_nav_result.store(OCR_NAV_NONE, std::memory_order_release);
                std::cout << "[Branch] state -> BRANCH_DETECTED" << std::endl;
            }
        }
        else
        {
            sign_detect_count_ = 0;
        }
    }
    else
    {
        sign_lost_count_++;
        if (sign_lost_count_ >= 3)
        {
            sign_detect_count_ = 0;
        }
    }
}

void Branch::run_branch(const std::vector<PredictResult> &predict_result, const float speed)
{
    // 检测本帧是否识别到岔路标志
    bool detected_this_frame = false;
    int branch_sign_width = 0;
    int bx1 = 0, by1 = 0, bx2 = 0, by2 = 0;
    for (const auto &obj : predict_result)
    {
        if (obj.class_id == CLASS_ID_BRANCHSIGN && obj.score >= confidence_threshold)
        {
            detected_this_frame = true;
            bx1 = obj.x1;
            by1 = obj.y1;
            bx2 = obj.x2;
            by2 = obj.y2;
            branch_sign_width = abs(obj.x1 - obj.x2);
            break;
        }
    }

    // 分叉形状判定
    multi_contour_count = (int)branch_rows_.size();

    if (branch_rows_.size() >= branch_min_rows)
    {
        // 从底部向上排序
        sort(branch_rows_.begin(), branch_rows_.end(),
             [](const BranchRow &a, const BranchRow &b)
             { return a.row > b.row; });

        // is_bell_shape钟形判断
        int n = (int)branch_rows_.size();
        bool bell = false;
        if (n >= branch_min_rows)
        {
            int peak_idx = 0;
            int peak_val = branch_rows_[0].pixel_count;
            for (int i = 1; i < n; ++i)
            {
                if (branch_rows_[i].pixel_count > peak_val)
                {
                    peak_val = branch_rows_[i].pixel_count;
                    peak_idx = i;
                }
            }
            if (peak_idx >= 1 && peak_idx <= n - 2)
            {
                int edge_avg = (branch_rows_[0].pixel_count + branch_rows_[n - 1].pixel_count) / 3;
                if (peak_val >= edge_avg * 3 && peak_val >= branch_min_pixels * 1.5)
                {
                    int violations = 0;
                    for (int i = 1; i <= peak_idx; ++i)
                    {
                        if (branch_rows_[i].pixel_count < branch_rows_[i - 1].pixel_count)
                            violations++;
                    }
                    if (violations <= peak_idx / 3 + 1)
                    {
                        violations = 0;
                        for (int i = peak_idx + 1; i < n; ++i)
                        {
                            if (branch_rows_[i].pixel_count > branch_rows_[i - 1].pixel_count)
                                violations++;
                        }
                        if (violations <= (n - peak_idx - 1) / 3 + 1)
                            bell = true;
                    }
                }
            }
        }
        has_branch = bell;
    }
    else
    {
        has_branch = false;
    }

    // ============ 岔路状态机 ============

    if (current_state_ == BRANCH_DETECTED)
    {
        // 计算当前路牌面积
        float current_area = 0.0f;
        if (detected_this_frame)
        {
            current_area = (float)abs((bx2 - bx1) * (by2 - by1));
            detected_lost_cnt = 0;
        }
        else
        {
            detected_lost_cnt ++;
        }

        if (detected_lost_cnt > 3)
        {
            current_area = 0.0f;
            current_state_ = BRANCH_NONE;
        }

        bool edges_clear = detected_this_frame &&
                           (bx1 > branch_sign_edge_x_margin * 3.5) &&
                           (by1 > branch_sign_edge_y_margin) &&
                           (IMAGE_W - bx2) > branch_sign_edge_x_margin * 3.5 &&
                           (IMAGE_H - by2) > branch_sign_edge_y_margin &&
                           (bx2 - bx1) > branch_sign_width_threshold;

        if (edges_clear)
        {
            // 首次满足 edges_clear → 记录初始面积（作为 AWAIT 面积比的基准）
            if (!initial_area_set_)
            {
                initial_sign_area_ = current_area;
                initial_area_set_ = true;
                std::cout << "[Branch] DETECTED initial sign area = " << initial_sign_area_
                          << " (bbox=" << bx1 << "," << by1 << "," << bx2 << "," << by2 << ")" << std::endl;
            }

            // 面积达到初始面积的 x2 倍 → 进入 BRANCH_AWAIT（不触发 OCR，等速度降下再触发）
            if (initial_area_set_ && current_area >= initial_sign_area_ * branch_sign_area_ratio_x2)
            {
                current_state_ = BRANCH_AWAIT;
                state_frame_count_ = 0;
                sign_lost_count_ = 0;
                g_ocr_nav_result.store(OCR_NAV_NONE, std::memory_order_release);
                std::cout << "[Branch] state -> BRANCH_AWAIT (area=" << current_area
                          << " initial=" << initial_sign_area_
                          << " ratio=" << (current_area / initial_sign_area_) << ")" << std::endl;
            }
            else
                detected_timeout_cnt++;
        }
        else
        {
            detected_timeout_cnt++;
        }

        if (detected_timeout_cnt > 350)
        {
            std::cerr << "[Branch] BRANCH_DETECTED timeout, fallback NONE" << std::endl;
            current_state_ = BRANCH_NONE;
            state_frame_count_ = 0;
            g_ocr_nav_result.store(OCR_NAV_NONE, std::memory_order_release);
        }
    }
    else if (current_state_ == BRANCH_AWAIT)
    {
        // 等待 OCR / API 返回结果
        int nav = g_ocr_nav_result.load(std::memory_order_acquire);

        if (nav == OCR_NAV_STRAIGHT || nav == OCR_NAV_RIGHT)
        {
            // API 已返回 → 进入 BRANCH_FETCH（车起步加速）
            current_state_ = BRANCH_FETCH;
            state_frame_count_ = 0;
            std::cout << "[Branch] state -> BRANCH_FETCH (API result: "
                      << (nav == OCR_NAV_STRAIGHT ? "STRAIGHT" : "RIGHT") << ")" << std::endl;
        }
        else
        {
            // 车速降至阈值 → 触发 OCR（仅触发一次）
            if (speed <= branch_ocr_speed && nav == OCR_NAV_NONE)
            {
                g_ocr_nav_result.store(OCR_NAV_NONE, std::memory_order_release);
                g_ocr_trigger.store(true, std::memory_order_release);
                std::cout << "[Branch] AWAIT speed OK, OCR triggered (speed=" << speed << ")" << std::endl;
            }

            // OCR 运行中 / 未返回，超时保护（~5秒 @30fps）
            state_frame_count_++;
            if (state_frame_count_ > 350)
            {
                std::cerr << "[Branch] OCR timeout, fallback STRAIGHT" << std::endl;
                g_ocr_nav_result.store(OCR_NAV_STRAIGHT, std::memory_order_release);
                current_state_ = BRANCH_FETCH;
                state_frame_count_ = 0;
            }
        }
    }
    else if (current_state_ == BRANCH_FETCH)
    {
        // 根据 OCR 结果分支：直行 → STRAIGHT，右转 → TURNRIGHT
        int nav = g_ocr_nav_result.load(std::memory_order_acquire);

        if (nav == OCR_NAV_STRAIGHT)
        {
            current_state_ = BRANCH_STRAIGHT;
            state_frame_count_ = 0;
            std::cout << "[Branch] state -> BRANCH_STRAIGHT" << std::endl;
        }
        else // OCR_NAV_RIGHT
        {
            current_state_ = BRANCH_TURNRIGHT;
            state_frame_count_ = 0;
            sign_lost_count_ = 0;
            std::cout << "[Branch] state -> BRANCH_TURNRIGHT" << std::endl;
        }
    }
    else if (current_state_ == BRANCH_TURNRIGHT)
    {
        // 右转预备：道路拟合已切为右侧，等待岔路标志消失 → 进入 BRANCH_IN
        if (!detected_this_frame)
        {
            sign_lost_count_++;
            if (sign_lost_count_ >= branch_in_frame_threshold)
            {
                current_state_ = BRANCH_IN;
                state_frame_count_ = 0;
                branch_in_count_ = 0;
                sign_lost_count_ = 0;
                std::cout << "[Branch] state -> BRANCH_IN" << std::endl;
            }
        }
        else
        {
            sign_lost_count_ = 0;
        }
    }
    else if (current_state_ == BRANCH_STRAIGHT)
    {
        // 直行加速至正常速度 → 回到 BRANCH_NONE（帧计数超时保护）
        state_frame_count_++;
        if (state_frame_count_ >= branch_none_frame_threshold + 20)
        {
            current_state_ = BRANCH_NONE;
            state_frame_count_ = 0;
            g_ocr_nav_result.store(OCR_NAV_NONE, std::memory_order_release);
            std::cout << "[Branch] STRAIGHT done, back to NONE" << std::endl;
        }
    }
    else if (current_state_ == BRANCH_IN)
    {
        // 检测到T字路口 → BRANCH_OUT
        if (is_t_junction)
        {
            current_state_ = BRANCH_OUT;
            state_frame_count_ = 0;
            branch_in_count_ = 0;
        }
    }
    else if (current_state_ == BRANCH_OUT)
    {
        // T字路口特征消失计数 → BRANCH_OVER
        if (!is_t_junction)
        {
            no_fork_count_++;
            if (no_fork_count_ >= branch_over_frame_threshold)
            {
                current_state_ = BRANCH_OVER;
                state_frame_count_ = 0;
                no_fork_count_ = 0;
            }
        }
        // else
        // {
        //     no_fork_count_ = 0;
        // }
    }
    else if (current_state_ == BRANCH_OVER)
    {
        rectangular_count_++;
        if (rectangular_count_ >= branch_none_frame_threshold)
        {
            current_state_ = BRANCH_NONE;
            state_frame_count_ = 0;
            rectangular_count_ = 0;
        }
        state_frame_count_++;
    }
}