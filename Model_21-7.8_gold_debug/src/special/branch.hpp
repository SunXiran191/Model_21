#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include "../../res/include/aiget.hpp"

struct Config;

class Branch
{
public:
    Branch();
    void init(const Config &config);

    enum BranchState
    {
        BRANCH_NONE = 0,  // 无岔路，左→右巡线
        BRANCH_DETECTED,  // 发现岔路标志
        BRANCH_AWAIT,     // ocr模型识别，传入api等待结果
        BRANCH_FETCH,     // 已获取结果，车起步
        BRANCH_TURNRIGHT, // 右转预备，道路拟合切为右侧，等岔路标志消失→BRANCH_IN
        BRANCH_IN,        // 岔路标志丢失，已驶入岔路，右→左巡线
        BRANCH_OUT,       // 接近主路丁字路口，右→左巡线
        BRANCH_OVER,      // 已过丁字路口，等待掩码矩形化，右→左巡线
        BRANCH_STRAIGHT   // 直行，加速至正常速度后回NONE
    };

    struct BranchRow
    {
        int row;
        int pixel_count;
        int center_x;
        int left_edge_x;
    };
    std::vector<BranchRow> branch_rows_;

    void reset_frame();

    void check_branch(const std::vector<PredictResult> &predict_result);
    void run_branch(const std::vector<PredictResult> &predict_result, const float speed);

    // T字路口检测（LineTracker，传入每行边缘坐标）
    void detect_t_junction(const std::vector<std::pair<int, int>> &row_edges, int total_rows);

    BranchState get_state() const { return current_state_; }

    // 右→左扫描：TURNRIGHT/IN/OUT/OVER状态下换道
    bool scan_right_to_left() const
    {
        return current_state_ == BRANCH_TURNRIGHT ||
               current_state_ == BRANCH_IN ||
               current_state_ == BRANCH_OUT ||
               current_state_ == BRANCH_OVER;
    }

    bool branch_await = false;    
    bool has_branch;              // 是否检测到钟形分叉
    bool is_t_junction;           // 本帧是否检测到T字路口
    int multi_contour_count;      // 分叉有效行数
    int detected_timeout_cnt = 0; // 本帧是否检测到岔路

    float confidence_threshold;

    int branch_detect_frame_threshold; // NONE→DETECTED: 岔路标志累计帧数阈值
    int branch_in_frame_threshold;     // 岔路标志丢失→进入岔路帧数阈值
    int branch_out_frame_threshold;    // 检测到T字路口→BRANCH_OUT帧数阈值
    int branch_over_frame_threshold;   // T字路口消失→BRANCH_OVER帧数阈值
    int branch_none_frame_threshold;   // 回NONE帧数阈值
    int branch_ocr_speed;
    int branch_sign_width_threshold; // 岔路标志检测框最小宽度
    int branch_sign_edge_x_margin;     // 标志框各边距画面边缘的最小距离
    int branch_sign_edge_y_margin;     // 标志框各边距画面边缘的最小距离
    float branch_sign_area_ratio_x2; // 进入AWAIT面积比阈值

    // T字路口
    int branch_min_pixels = 8;         // 岔路最小像素数
    int branch_min_rows = 8;           // 岔路最小行数
    float t_junction_margin = 60.0f;   // 外推距离
    int t_junction_row_threshold = 60; // 超出行数阈值

private:
    BranchState current_state_;
    int state_frame_count_;
    int sign_detect_count_; // 岔路标志连续检测帧计数
    int sign_lost_count_;
    int detected_lost_cnt;
    int branch_in_count_;
    int no_fork_count_;
    int rectangular_count_;
    float initial_sign_area_; // 初始路牌面积
    bool initial_area_set_;   // 是否已记录初始面积
};
