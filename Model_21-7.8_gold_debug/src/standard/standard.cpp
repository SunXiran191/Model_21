#include "standard.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>

#include "../imgprocess/TrajectoryFitter.hpp"
#include "../thread/thread.hpp"

Scene_status::Scene_status() : GoldScene(false), CarScene(false), HumanScene(false), LightScene(false), GoScene(false) {}

bool Scene_status::all()
{
    return GoldScene || CarScene || HumanScene || LightScene;
}

Standard::Standard() : Standard(Config())
{
}

Standard::Standard(Config config) : objects_(config), human_(config)
{
    this->config_ = config;
    aim_distance_f = config.aim_distance_f;
    gold_fix_y = config.gold_fix_y;
    human_stop_threshold = config.human_stop_threshold;
    fuzzy_human_speed_.loadConfig(config); // 加载模糊控制器配置
    light_ = Light(config);
    go_stop_ = GoStop(config);
    tracker.init(config);
    tracker.branch_.init(config);
    general.set_save_dir(config.save_image_dir);
}

// ============================================================================
//               上位机流程：帧提取→巡线→采样→元素处理→偏差→速度
// ============================================================================
TaskData Standard::run(Produce &G_produce, float &run_speed, double run_fps)
{
    TaskData data;
    data.speed = run_speed;
    // 分段计时
    auto t_seg = std::chrono::steady_clock::now();
    auto mark = [&]()
    {
        auto n = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(n - t_seg).count();
        t_seg = n;
        return ms;
    };

    // *********************** 提取帧数据 ************************ //
    cv::Mat src_img, src_segmask, morph_segmask, mask;
    std::vector<PredictResult> predict_result;
    if (!extractFrameData(G_produce, src_img, src_segmask, predict_result))
        return data;
    double t_copy = mark();

    // *********************** AI检测状态更新 ************************ //
    if (1)
        trackstate = TrackState::TRACK_AI_MIDDLE;
    else
        trackstate = TrackState::TRACK_CV_MIDDLE;

    updateSceneStatus(predict_result, scene_status, &config_);
    double t_update = mark();

    // *********************** 巡线提取 ************************ //
    extractLanePoints(src_img, src_segmask, predict_result, morph_segmask, mask, G_produce, run_speed);
    double t_extract = mark();
    double t_fit = fit_ms_;

    // *********************** 等距采样 + 曲率计算 *********************** //
    float cx = 0.0f, cy = 0.0f;
    cv::Point2f car_base_ipm;
    int traj_sz = 0;
    bool traj_valid = computeTrajectory(cx, cy, car_base_ipm, traj_sz);
    double t_curve = mark();

    if (traj_valid)
    {
        // 弦高法曲率计算（更新 meanCurvature_）
        {
            std::vector<cv::Point2d> temp_double;
            dynamicAimDisCal(s_t_trackPoints_AI, temp_double);
        }
        // 拷贝 int 精度轨迹点供后续元素拟合/偏差计算使用
        d_s_t_trackPoints_AI.clear();
        d_s_t_trackPoints_AI.reserve(s_t_trackPoints_AI.size());
        for (const auto &pt : s_t_trackPoints_AI)
            d_s_t_trackPoints_AI.push_back(pt);

        // *********************** 特殊元素处理 ************************ //
        processSpecials(predict_result, src_img, traj_sz, run_speed, run_fps, data, car_base_ipm);
        double t_objects = mark();

        // *********************** 偏差计算 + 丢线恢复 ************************ //
        computeError(cx, cy, car_base_ipm, traj_sz, data);
        double t_error = mark();

        // *********************** 金币夹角触发检查 ************************ //
        gold_in_angle = 180.0f;
        if (objects_.gold_seq_y_max > objects_.gold_seq_y_min)
        {
            float aim_y = aim_point.y;
            if (aim_y >= objects_.gold_seq_y_min && aim_y <= objects_.gold_seq_y_max)
            {
                objects_.gold_angle_flag = true;
                // 遍历查找 aim_y 所在Y范围对应的角度
                for (const auto &info : objects_.gold_angles)
                {
                    if (aim_y >= info.y_low && aim_y <= info.y_high)
                    {
                        gold_in_angle = info.angle; // gold_in_angle为当前预瞄点位置的对应角度，偏离180°越大弯越急
                        break;
                    }
                }
            }
        }

        // *********************** 速度决策 *********************** //
        speedDecision(run_speed, data, car_base_ipm);

        // *********************** 性能统计 *********************** //
        printStats(t_copy, t_update, t_extract, t_fit, t_curve, t_objects, t_error);
    }
    else
    {
        // 轨迹无效时重置行人控制状态，防止残留上一帧的非NONE值
        human_.resetControlState();

        // 轨迹无效时执行丢线恢复
        if (lost_cnt_ > 40)
        {
            data.speed = 0;
            lost_cnt_ = 0;
            side_history_cnt_ = 0;
            side_history_idx_ = 0;
            data.track_side = 0;
        }
        else if (lost_cnt_ > 3)
        {
            data.speed = run_speed;
            data.lost_flag = true;
            if (side_history_cnt_ > 0)
            {
                float sum = 0.0f;
                for (int i = 0; i < side_history_cnt_; i++)
                    sum += side_history_[i];
                float avg_dx = sum / (float)side_history_cnt_;
                const float kSideDead = 5.0f;
                if (avg_dx > kSideDead)
                    data.track_side = 1;
                else if (avg_dx < -kSideDead)
                    data.track_side = -1;
                else
                    data.track_side = 0;
            }
        }
        speedDecision(run_speed, data, car_base_ipm);
    }

    if (light_.light_action == Light::ACTION_STOP)
    {
        data.speed = 0;
    }
    return data;
}

// ============================================================================
// 从 Produce 拷贝帧数据
// ============================================================================
bool Standard::extractFrameData(Produce &G_produce, cv::Mat &src_img,
                                cv::Mat &src_segmask,
                                std::vector<PredictResult> &predict_result)
{
    extern std::mutex mtx_produce;
    std::lock_guard<std::mutex> lock(mtx_produce);
    if (G_produce.img.empty())
        return false;
    src_img = G_produce.img;
    src_segmask = G_produce.seg_mask;
    predict_result = G_produce.predict_results;
    return true;
}

// ============================================================================
// 巡线提取（CV或AI分割,输出t_trackPoints_CV）
// ============================================================================
void Standard::extractLanePoints(const cv::Mat &src_img, const cv::Mat &src_segmask,
                                 const std::vector<PredictResult> &predict_result,
                                 cv::Mat &morph_segmask, cv::Mat &mask,
                                 Produce &G_produce, const float speed)
{
    if (trackstate == TrackState::TRACK_CV_MIDDLE)
    {
        std::vector<cv::Point> lane_points = tracker.ExtractArrows_CV(src_img, mask);
        if (lane_points.size() > 3)
        {
            trackPoints_CV.clear();
            trackPoints_CV = FitTrajectory_LOWESS(PTS_LINE_NUM, lane_points, src_img);
            // trackPoints_CV = FitTrajectory_Gauss(PTS_LINE_NUM, lane_points, src_img);
            t_trackPoints_CV.clear();
            for (size_t i = 0; i < trackPoints_CV.size(); i++)
            {
                cv::Point2f t_pt = transf((float)trackPoints_CV[i].x, (float)trackPoints_CV[i].y);
                if ((t_pt.x < IMAGE_W - 1 && t_pt.x > 0) && (t_pt.y < IMAGE_H - 1 && t_pt.y > 0))
                    t_trackPoints_AI.emplace_back(cvRound(t_pt.x), cvRound(t_pt.y));
            }
        }
        else
        {
            lost_cnt_++;
        }
        return;
    }
    else if (trackstate == TrackState::TRACK_AI_MIDDLE)
    {

        std::vector<cv::Point> path_t_branch;
        t_trackPoints_AI.clear();
        // 预分配容量
        t_trackPoints_AI.reserve(200);
        path_t_branch.reserve(200);

        if (!src_segmask.empty())
            img_process.Img_process_AI(src_segmask, morph_segmask);

        tracker.branch_.check_branch(predict_result);
        tracker.ExtractArrows_AI(morph_segmask, mask, t_trackPoints_AI, path_t_branch);

        if (tracker.branch_.get_state() != Branch::BRANCH_NONE)
            tracker.branch_.run_branch(predict_result, speed);

        if (tracker.branch_.scan_right_to_left())
            std::swap(t_trackPoints_AI, path_t_branch);

        // 将处理后的 mask 写回 Produce 供可视化使用
        {
            extern std::mutex mtx_produce;
            std::lock_guard<std::mutex> lock(mtx_produce);
            G_produce.morph_mask = mask.clone();
            G_produce.ipm_mask = G_produce.morph_mask;
        }

        // LOWESS 拟合
        if (t_trackPoints_AI.size() > 3)
        {
            auto t0 = std::chrono::steady_clock::now();
            t_trackPoints_AI = FitTrajectory_LOWESS(PTS_LINE_NUM, t_trackPoints_AI, morph_segmask);
            // t_trackPoints_AI = FitTrajectory_Gauss(PTS_LINE_NUM, t_trackPoints_AI, morph_segmask);
            auto t1 = std::chrono::steady_clock::now();
            fit_ms_ = std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        else
        {
            lost_cnt_++;
            return;
        }

        // 记录车相对赛道中线的左右位置（3帧环形缓冲防抖动）
        cv::Point2f car_ipm = transf(IMAGE_W / 2.0f, IMAGE_H * 0.95f);
        int best_i = 0;
        float best_d2 = 1e10f;
        for (int i = 0; i < (int)t_trackPoints_AI.size(); i++)
        {
            float dx = (float)t_trackPoints_AI[i].x - car_ipm.x;
            float dy = (float)t_trackPoints_AI[i].y - car_ipm.y;
            float d2 = dx * dx + dy * dy;
            if (d2 < best_d2)
            {
                best_d2 = d2;
                best_i = i;
            }
        }
        int r = 2;
        int lo = std::max(0, best_i - r);
        int hi = std::min((int)t_trackPoints_AI.size() - 1, best_i + r);
        float avg_x = 0.0f;
        for (int i = lo; i <= hi; i++)
            avg_x += (float)t_trackPoints_AI[i].x;
        avg_x /= (float)(hi - lo + 1);

        float dx = car_ipm.x - avg_x;
        side_history_[side_history_idx_] = dx;
        side_history_idx_ = (side_history_idx_ + 1) % 3;
        if (side_history_cnt_ < 3)
            side_history_cnt_++;

        // 诊断输出
        Branch::BranchState st = tracker.branch_.get_state();
        const char *state_str = "NONE";
        switch (st)
        {
        case Branch::BRANCH_NONE:
            state_str = "NONE";
            break;
        case Branch::BRANCH_DETECTED:
            state_str = "DETECTED";
            break;
        case Branch::BRANCH_AWAIT:
            state_str = "AWAIT";
            break;
        case Branch::BRANCH_FETCH:
            state_str = "FETCH";
            break;
        case Branch::BRANCH_TURNRIGHT:
            state_str = "TURNRIGHT";
            break;
        case Branch::BRANCH_IN:
            state_str = "IN";
            break;
        case Branch::BRANCH_OUT:
            state_str = "OUT";
            break;
        case Branch::BRANCH_OVER:
            state_str = "OVER";
            break;
        case Branch::BRANCH_STRAIGHT:
            state_str = "STRAIGHT";
            break;
        default:
            break;
        }
        if (std::strcmp(state_str, "NONE") != 0 || tracker.branch_.has_branch)
        {
            int nav = g_ocr_nav_result.load(std::memory_order_relaxed);
            const char *ocr_str =
                (nav == OCR_NAV_PENDING) ? " [OCR pending]" : (nav == OCR_NAV_STRAIGHT) ? " [OCR→直行]"
                                                          : (nav == OCR_NAV_RIGHT)      ? " [OCR→右转]"
                                                                                        : "";
            printf("🔀 Branch state=%s has_branch=%d%s\n",
                   state_str, (int)tracker.branch_.has_branch, ocr_str);
        }
    }
}

// ============================================================================
// 等距采样 + 曲率计算
// ============================================================================
bool Standard::computeTrajectory(float &cx, float &cy, cv::Point2f &car_base_ipm, int &traj_sz)
{
    // 计算车在IPM中的位置（底边中点）
    car_base_ipm = transf(IMAGE_W / 2.0f, IMAGE_H * 0.95f);
    cx = car_base_ipm.x;
    cy = car_base_ipm.y;

    if (t_trackPoints_AI.size() <= 3)
    {
        traj_sz = 0;
        return false;
    }

    // 拷贝到 s_t_trackPoints_AI
    s_t_trackPoints_AI.clear();
    s_t_trackPoints_AI.reserve(t_trackPoints_AI.size());
    for (const auto &pt : t_trackPoints_AI)
        s_t_trackPoints_AI.emplace_back(pt.x, pt.y);

    int cv_sz = static_cast<int>(s_t_trackPoints_AI.size());

    // 找最近点
    float min_dist = 1e7f;
    int begin_id = -1;
    for (int i = 0; i < cv_sz; i++)
    {
        float dx = s_t_trackPoints_AI[i].x - cx;
        float dy = s_t_trackPoints_AI[i].y - cy;
        float dist = sqrt(dx * dx + dy * dy);
        if (dist < min_dist)
        {
            min_dist = dist;
            begin_id = i;
        }
    }
    begin_id = clip(begin_id, 0, cv_sz - 1);

    if (begin_id < 0 || cv_sz - begin_id < 3)
    {
        traj_sz = 0;
        return false;
    }

    cx = s_t_trackPoints_AI[begin_id].x;
    cy = s_t_trackPoints_AI[begin_id].y;

    // 等距采样
    std::vector<cv::Point> temp_center;
    temp_center.reserve(cv_sz - begin_id); // 预分配
    for (int i = begin_id; i < cv_sz; i++)
    {
        temp_center.emplace_back(s_t_trackPoints_AI[i].x, s_t_trackPoints_AI[i].y);
    }

    int temp_sz = static_cast<int>(temp_center.size());
    s_t_trackPoints_AI.clear();
    int sampled_sz = 0;
    tracker.resample_points(temp_center, temp_sz, s_t_trackPoints_AI, sampled_sz,
                            RESAMPLEDIST * PIXPERMETER);

    traj_sz = static_cast<int>(s_t_trackPoints_AI.size());
    return (traj_sz >= 3);
}

// ============================================================================
// 子方法 4：特殊元素处理（灯光/障碍物/金币/行人/停车）
// ============================================================================
void Standard::processSpecials(const std::vector<PredictResult> &predict_result, cv::Mat &src_img, int &traj_sz,
                               float &run_speed, double run_fps, TaskData &data, const cv::Point2f &car_base_ipm)
{
    Branch::BranchState branch_state = tracker.branch_.get_state();

    // // ---- 灯光检测与决策 ----
    // light_.check_light_color(predict_result);
    // if (light_.light_color != Light::COLOR_NONE)
    // {
    //     light_.run_light(predict_result, t_light_point, t_zebra_point,
    //                      static_cast<int>(run_fps), run_speed, run_speed);
    //     scene_status_e = Light_status;
    // }
    // else
    // {
    //     scene_status_e = Normal_status;
    // }

    // ---- 障碍物/金币处理 ----
    objects_.processAllObjects(src_img, t_gold_point, t_car_point,
                               d_s_t_trackPoints_AI, traj_sz);

    // ---- STOP/GO 标志牌轨迹修正（遮挡道路导致语义分割巡线错误） ----
    if (branch_state == Branch::BRANCH_NONE)
    {
        if ((!t_stop_point.empty() || !t_go_point.empty()) && traj_sz >= 3)
        {
            go_stop_.processGoStop(t_stop_point, t_go_point,
                                   d_s_t_trackPoints_AI, traj_sz);
        }
    }

    // ---- 行人状态机处理 ----
    // 计算车实际位置相对拟合中线起点的横向偏移，用于补偿道路弯曲
    float car_track_offset = 0.0f;
    if (!d_s_t_trackPoints_AI.empty())
    {
        car_track_offset = car_base_ipm.x - d_s_t_trackPoints_AI[0].x;
    }
    human_.run_human(t_human_point, d_s_t_trackPoints_AI, traj_sz, car_track_offset);

    // ---- 启动逻辑 ----

    // ---- 停车逻辑 ----
    if (!t_stop_point.empty() && t_stop_point[0].y < IMAGE_H - 200)
    {
        if (stop_delay_cnt_ < 0)
            stop_delay_cnt_ = 50;
    }
    if (stop_delay_cnt_ > 0)
    {
        stop_delay_cnt_--;
    }
    else if (stop_delay_cnt_ == 0)
    {
        data.speed = 0.0f;
        stop_delay_cnt_ = -1;
    }
}

// ============================================================================
// 偏差计算 + 丢线恢复
// ============================================================================
void Standard::computeError(float cx, float cy, const cv::Point2f &car_base_ipm,
                            int traj_sz, TaskData &data)
{
    // 预瞄点选择

    // ------------------------------------------------------------------------
    // 优化点 1: 【正向反馈】利用刚才算出的“弦高(meanCurvature_)”来动态缩放预瞄距离
    // ------------------------------------------------------------------------
    // 假设直道弦高很小(<10)，急弯(R1300/R1500)弦高较大(可能在 50~150 像素之间)
    // 根据实际情况，你可以把这个 MAX_BEND 调成你在最急的弯道测出来的最大 meanCurvature_ 值
    const float MAX_BEND = 100.0f;
    float bend_ratio = clipf(meanCurvature_ / MAX_BEND, 0.0f, 1.0f);

    // 弯道越急(bend_ratio趋近1)，看越近(最近0.6倍)；直道(bend_ratio=0)，看越远(1.0倍)
    float lookahead_scale = 1.0f - 0.4f * bend_ratio;
    double target_dist = lookahead_scale * config_.aim_distance_f * PIXPERMETER;

    //----------------------------------------------------------------------------//

    // double target_dist = clipf((1 - pre_error / 43.0f * pre_error / 43.0f), 0.6, 1.0) * config_.aim_distance_f * PIXPERMETER;
    // double target_dist = config_.aim_distance_f * PIXPERMETER;

    Branch::BranchState branch_state = tracker.branch_.get_state();
    if (branch_state == Branch::BRANCH_TURNRIGHT)
    {
        target_dist = config_.aim_distance_branch_f * PIXPERMETER;
    }
    else if (branch_state == Branch::BRANCH_IN || branch_state == Branch::BRANCH_OUT)
    {
        target_dist = config_.aim_distance_branch_f * PIXPERMETER * 1.5f;
    }

    aim_index_far = traj_sz - 1;
    for (int i = 1; i < traj_sz; i++)
    {
        double dx = d_s_t_trackPoints_AI[i].x - cx;
        double dy = cy - d_s_t_trackPoints_AI[i].y;
        double dn = sqrt(dx * dx + dy * dy);
        if (dn >= target_dist)
        {
            double dxp = d_s_t_trackPoints_AI[i - 1].x - cx;
            double dyp = cy - d_s_t_trackPoints_AI[i - 1].y;
            double dnp = sqrt(dxp * dxp + dyp * dyp);
            aim_index_far = (fabs(dnp - target_dist) <= fabs(dn - target_dist)) ? (i - 1) : i;
            break;
        }
    }

    aim_point = d_s_t_trackPoints_AI[aim_index_far];

    // Pure Pursuit (纯追踪) 算法
    float dx = d_s_t_trackPoints_AI[aim_index_far].x - cx;
    float dy = cy - d_s_t_trackPoints_AI[aim_index_far].y;
    float Ld_sq = dx * dx + dy * dy;
    Ld_sq = std::max(Ld_sq, 1.0f); // 防止除零异常

    // 纯追踪方向盘转角公式: δ = arctan(2 * L * dx / Ld^2)
    float wb_px = static_cast<float>(wheelbase * PIXPERMETER); // 车辆轴距(像素)
    float error = -atan2f(2.0f * wb_px * dx, Ld_sq) * 180.0f / PI;

    const float X_ERROR_SAT = 240.0f;
    float start_cnt = clipf(0.05f * static_cast<float>(d_s_t_trackPoints_AI.size()), 0.0f,
                            static_cast<float>(d_s_t_trackPoints_AI.size()));
    float x_error = clipf(car_base_ipm.x - d_s_t_trackPoints_AI[static_cast<int>(start_cnt)].x,
                          -X_ERROR_SAT, X_ERROR_SAT);

    // 动态预瞄点角度误差（用于下一帧的 pre_error）
    float pre_dx = s_t_trackPoints_AI[aim_index_far].x - cx;
    float pre_dy = cy - s_t_trackPoints_AI[aim_index_far].y + wb_px;
    float last_error = -atan2f(pre_dx, pre_dy) * 180.0f / PI;

    assert(!isnan(error));

    if (branch_state == Branch::BRANCH_TURNRIGHT)
        error *= 1.5f;

    data.error = error;
    data.x_error = x_error;
    pre_error = last_error;
    // 巡线成功，重置丢线计数
    lost_cnt_ = 0;
    data.lost_flag = false;
    data.track_side = 0;
}

// ============================================================================
// 速度决策 + 密集目标减速
// ============================================================================

void Standard::speedDecision(float &ctrl_speed, TaskData &data, cv::Point2f &car_base_ipm)
{

    max_speed_ = config_.speedHigh;
    // human_distance_thresh_ = config_.HUMAN_DISTANCE_THRESH;
    const float kStep = 0.35f; // 每帧最大速度变化量
    Human::ControlState human_ctrl = human_.getControlState();
    Branch::BranchState branch_state = tracker.branch_.get_state();
    float target;
    // 检测行人避让刚结束 → 触发冲刺加速
    if (prev_human_ctrl_ != Human::ControlState::NONE &&
        human_ctrl == Human::ControlState::NONE)
    {
        pedestrian_recovery_cnt_ = 15; // 冲刺约 0.5 秒 (@30fps)
    }
    prev_human_ctrl_ = human_ctrl;

    // ============ 行人模糊速度控制 ============
    float offset_ratio = human_.getClosestOffsetRatio();

    float distance_ratio = static_cast<float>(human_.human_dist_y) / IMAGE_H;
    float speed_scale = fuzzy_human_speed_.compute(static_cast<int>(human_ctrl), offset_ratio, distance_ratio);
    // printf("🚶‍♂️ human_ctrl=%d offset_ratio=%.2f distance_ratio=%.2f speed_scale=%.2f\n",
    //       (int)human_ctrl, offset_ratio, distance_ratio, speed_scale);
    target = max_speed_ * speed_scale;
    // printf("scene_status.HumanScene=%d human_ctrl=%d\n", scene_status.HumanScene, (int)human_ctrl);

    if (scene_status.HumanScene && human_ctrl != Human::ControlState::SPEEDUP && human_ctrl != Human::ControlState::GO)
    {
        dist_human_reach = 0;
        int human_dist_y = abs(car_base_ipm.y - t_human_point[0].y);
        int human_dist_x = abs(t_human_point[0].x - car_base_ipm.x);
        dist_human_reach = sqrt(human_dist_x * human_dist_x + human_dist_y * human_dist_y);
        // printf("🚶‍♂️ Human detected, distance=%.2f px\n", dist_human_reach);
        ctrl_speed -= kStep * (1 - speed_scale);
        if (ctrl_speed > max_speed_ * dist_human_reach / 460.0f)
            ctrl_speed = max_speed_ * dist_human_reach / 460.0f;
        // 安全兜底：即使上述条件不触发，也不能超过 max_speed_
        if (ctrl_speed > max_speed_)
            ctrl_speed = max_speed_;
        if (human_ctrl == Human::ControlState::NONE)
        {
            return;
        }
    }

    if (human_ctrl != Human::ControlState::NONE)
    {
        // 平滑逼近目标速度：减速比加速更快（安全优先）
        if (human_ctrl == Human::ControlState::STOP)
        {
            ctrl_speed = 0.0f; // 行人完全阻挡时，目标速度为0
        }
        else if (human_ctrl == Human::ControlState::SLOW)
        {
            ctrl_speed -= kStep * (1 - speed_scale); // 快速减速
            if (ctrl_speed < target)
                ctrl_speed = target;
        }
        else if (human_ctrl == Human::ControlState::SPEEDUP)
        {
            ctrl_speed += kStep; // 正常加速
            if (ctrl_speed > max_speed_)
                ctrl_speed = max_speed_;
        }
        else if (human_ctrl == Human::ControlState::GO)
        {
            ctrl_speed += kStep; // 正常加速
            if (ctrl_speed > max_speed_)
                ctrl_speed = max_speed_;
        }

        // 行人控制激活时，跳过下方的通用速度逻辑
        return;
    }

    // 岔路牌减速：scene_status 已在 updateSceneStatus 中由 AI 检测结果设置，无需重复遍历
    if (branch_state != Branch::BRANCH_NONE)
    {
        if (branch_state == Branch::BRANCH_DETECTED)
        {
            // 1.6m//
            ctrl_speed -= kStep * 0.6f;
            if (ctrl_speed <= 0.4f)
                ctrl_speed = 0.4f;
            // 1.4m//
            //  ctrl_speed -= kStep * 0.7f;
            //  if (ctrl_speed <= 0.4f)
            //      ctrl_speed = 0.4f;
        }
        if (branch_state == Branch::BRANCH_AWAIT)
        {
            ctrl_speed -= kStep * 0.1f;
            if (ctrl_speed <= 0.1f)
                ctrl_speed = 0;
        }
        if (branch_state == Branch::BRANCH_FETCH || branch_state == Branch::BRANCH_IN || branch_state == Branch::BRANCH_OUT || branch_state == Branch::BRANCH_OVER || branch_state == Branch::BRANCH_TURNRIGHT || branch_state == Branch::BRANCH_STRAIGHT)
        {
            ctrl_speed += kStep * 0.2f;
            if (branch_state == Branch::BRANCH_TURNRIGHT)
                max_speed_ = 0.8f * max_speed_;
            if (ctrl_speed > max_speed_)
                ctrl_speed = max_speed_;
            return;
        }
        return;
    }

    // 原有速度逻辑：统一加速 + 统一上限，防止任何分支速度失控
    {
        ctrl_speed += 2 * kStep;

        // 根据误差设置不同的目标上限
        float speed_limit = max_speed_;
        // 1.6//
        if (fabs(data.error) <= 15.0f)
        {
            str_speed_up_cnt_++;
        }
        else
            str_speed_up_cnt_ = 0;
        if (str_speed_up_cnt_ > 5)
        {
            speed_limit = config_.speedHigh * 1.15f;
            str_speed_up_cnt_ = 0;
        }
        //         if (fabs(data.error) <= 15.0f)
        // {
        //     ctrl_speed += 0.01f;
        //     speed_limit = config_.speedHigh * 1.15f;
        // }

        // ── 金币角度限速 ──
        // gold_in_angle 范围 0~180°：越小 → 金币排布越弯曲（弯越急）→ 需要更低的速度
        // 越大 → 越接近直线 → 不需要减速
        // printf("gold_in_angle=%.2f\n", gold_in_angle);
        if (gold_in_angle < 175.0f) // 夹角 >150° 视为近似直线，不限速
        {
            const float kMaxDecelRatio = 0.95f;              // 最大减速比例（0°时速度上限降至 max_speed_ * 0.5）
            float inv_ratio = 1.0f - gold_in_angle / 175.0f; // 0°→1, 170°→0
            // 平方映射：小角度（急弯）减速明显，接近 170° 时几乎无影响
            float gold_limit = max_speed_ * (1.0f - kMaxDecelRatio * inv_ratio * inv_ratio);
            if (speed_limit > gold_limit)
                speed_limit = gold_limit;
        }

        // 统一钳位，确保不超任何上限
        if (ctrl_speed > speed_limit)
            ctrl_speed = speed_limit;
        // 速度下限保护（防止减速到负值或停车）
        if (ctrl_speed < 0.15f)
            ctrl_speed = 0.15f;
        return;
    }
}

// void Standard::speedDecision(float &ctrl_speed, TaskData &data, cv::Point2f &car_base_ipm)
// {
//     if (max_speed_ <= 0.0f && ctrl_speed > 0.0f)
//         max_speed_ = ctrl_speed;
//     if (config_.speedHigh > max_speed_)
//         max_speed_ = config_.speedHigh;

//     const float kStep = 0.35f; // 每帧最大速度变化量
//     Human::ControlState human_ctrl = human_.getControlState();
//     Branch::BranchState branch_state = tracker.branch_.get_state();

//     // 检测行人避让刚结束 → 触发冲刺加速
//     if (prev_human_ctrl_ != Human::ControlState::NONE &&
//         human_ctrl == Human::ControlState::NONE)
//     {
//         pedestrian_recovery_cnt_ = 15; // 冲刺约 0.5 秒 (@30fps)
//     }
//     prev_human_ctrl_ = human_ctrl;

//     // ============ 行人模糊速度控制 ============
//     if (human_ctrl != Human::ControlState::NONE)
//     {
//         float offset_ratio = human_.getClosestOffsetRatio();

//         float distance_ratio = static_cast<float>(human_.human_dist_y) / IMAGE_H;
//         float speed_scale = fuzzy_human_speed_.compute(
//             static_cast<int>(human_ctrl), offset_ratio, distance_ratio);
//         // printf("🚶‍♂️ human_ctrl=%d offset_ratio=%.2f distance_ratio=%.2f speed_scale=%.2f\n",
//         //       (int)human_ctrl, offset_ratio, distance_ratio, speed_scale);
//         float target = m        printf("gold_in_angle=%.2f\n", gold_in_angle);
// ax_speed_ * speed_scale;
//          / 平滑逼近目标速度：减速比加速更快（安全优先）
//                  const float kMaxDecelRatio = 0.70f;        // 最大减速比例（0°时速度上限降至 max_speed_ * 0.5）
//             if (ctrl_speed < target)
//                 c rl_speed = target;
//         }
//         else if (ctrl_speed < target - kStep)
//         {
//             ctrl_speed += kStep * 100.0f; // 正常加速
//             if (ctrl_speed > target)
//                 ctrl_speed = target;
//         }
//         else
//         {
//             ctrl_speed = target;
//         }
//         // 行人控制激活时，跳过下方的通用速度逻辑
//         return;
//     }
//     // if(branch_state == Branch::BRANCH_STRAIGHT)
//     // {
//     //     ctrl_speed += kStep ; // 正常加速
//     //         if (ctrl_speed > max_speed_)
//     //             ctrl_speed = max_speed_;
//     //     return;
//     // }

//     // 行人避让结束后的冲刺（短暂超速，上限 125% max_speed_）
//     if (pedestrian_recovery_cnt_ > 0)
//     {
//         pedestrian_recovery_cnt_--;
//         const float sprint_max = max_speed_ * 1.05f;
//         ctrl_speed += kStep * 20.0f;
//         if (ctrl_speed > sprint_max)
//             ctrl_speed = sprint_max;
//     }
//     else if (scene_status.HumanScene && human_ctrl == Human::ControlState::NONE)
//     {
//         dist_human_reach = 0;
//         int human_dist_y = abs(car_base_ipm.y - t_human_point[0].y);
//         int human_dist_x = abs(t_human_point[0].x - car_base_ipm.x);
//         dist_human_reach = sqrt(human_dist_x * human_dist_x + human_dist_y * human_dist_y);
//         printf("🚶‍♂️ Human detected, distance=%.2f px\n", dist_human_reach);
//         ctrl_speed -= kStep;
//         if (ctrl_speed < 0)
//             ctrl_speed = 0;
//     }

//     // 岔路牌减速：scene_status 已在 updateSceneStatus 中由 AI 检测结果设置，无需重复遍历
//     if (branch_state != Branch::BRANCH_NONE)
//     {
//         if (branch_state == Branch::BRANCH_DETECTED || branch_state == Branch::BRANCH_AWAIT)
//         {
//             ctrl_speed -= kStep * 0.6f;
//             if (ctrl_speed < 0.6f)
//                 ctrl_speed = 0.1f;
//             return;
//         }
//         if (branch_state == Branch::BRANCH_FETCH || branch_state == Branch::BRANCH_IN || branch_state == Branch::BRANCH_OUT || branch_state == Branch::BRANCH_OVER || branch_state == Branch::BRANCH_TURNRIGHT || branch_state == Branch::BRANCH_STRAIGHT)
//         {
//             ctrl_speed += kStep * 0.8f;
//             if (ctrl_speed > max_speed_)
//                 ctrl_speed = max_speed_;
//             return;
//         }
//         return;
//     }

//     // 原有速度逻辑
//     if (ctrl_speed == 0.0f)
//     {
//         ctrl_speed += 2 * kStep;
//         if (max_speed_ > 0.0f && ctrl_speed > max_speed_)
//             ctrl_speed = max_speed_;
//         return;
//     }

//     else
//     {
//         ctrl_speed += 2 * kStep;
//         if (ctrl_speed > max_speed_)
//             ctrl_speed = max_speed_;
//     }

//     // 密集目标强制减速
//     // if (data.speed > 0.0f)
//     // {
//     //     int dense_cnt = 0;
//     //     int y_low = config_.dense_y_low;
//     //     int y_high = config_.dense_y_high;
//     //     for (const auto &pt : t_car_point)
//     //         if (pt.y >= y_low && pt.y <= y_high)
//     //             dense_cnt++;
//     //     for (const auto &pt : t_human_point)
//     //         if (pt.y >= y_low && pt.y <= y_high)
//     //             dense_cnt++;
//     //     for (const auto &pt : t_gold_point)
//     //         if (pt.y >= y_low && pt.y <= y_high)
//     //             dense_cnt++;
//     //     if (dense_cnt > 3)
//     //         data.speed *= config_.dense_speed;
//     // }
// }

// ============================================================================
// 子方法 7：性能统计（每秒输出一次）
// ============================================================================
void Standard::printStats(double copy_ms, double update_ms, double extract_ms, double fit_ms,
                          double curve_ms, double objects_ms, double error_ms)
{
    prof_copy_ += copy_ms;
    prof_update_ += update_ms;
    prof_extract_ += extract_ms;
    prof_fit_ += fit_ms;
    prof_curve_ += curve_ms;
    prof_objects_ += objects_ms;
    prof_error_ += error_ms;
    ++prof_frames_;

    auto t_now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t_now - prof_last_).count();
    // if (elapsed >= 1.0)
    // {
    //     double d = prof_frames_ > 0 ? (double)prof_frames_ : 1.0;
    //     printf("[run detail] copy=%.2f update=%.2f extract=%.2f fit=%.2f curve=%.2f objects=%.2f error=%.2f | total=%.2fms\n",
    //            prof_copy_ / d, prof_update_ / d, prof_extract_ / d, prof_fit_ / d,
    //            prof_curve_ / d, prof_objects_ / d, prof_error_ / d,
    //            (prof_copy_ + prof_update_ + prof_extract_ + prof_fit_ + prof_curve_ + prof_objects_ + prof_error_) / d);
    //     prof_frames_ = 0;
    //     prof_copy_ = prof_update_ = prof_extract_ = prof_fit_ = prof_curve_ = prof_objects_ = prof_error_ = 0.0;
    //     prof_last_ = t_now;
    // }
}

void Standard::updateSceneStatus(const std::vector<PredictResult> &frame, Scene_status &scene_status, const Config *config)
{

    float confidence_threshold = (config != nullptr) ? config->confidence_threshold : 0.44f;

    // 重置所有标志
    scene_status.GoldScene = false;
    scene_status.CarScene = false;
    scene_status.HumanScene = false;
    scene_status.LightScene = false;
    scene_status.BranchScene = false;
    scene_status.SpeedLimitScene = false;
    scene_status.StopScene = false;
    scene_status.GoScene = false;

    // 收集所有置信度达标的目标，按 y 坐标从大到小排序（底部→顶部，由近到远）
    std::vector<PredictResult> all_objects;
    all_objects.reserve(frame.size());

    for (const auto &obj : frame)
    {
        if (obj.score > confidence_threshold)
        {
            all_objects.push_back(obj);
            int cid = obj.class_id;
            // 更新 Scene_status 标志
            switch (cid)
            {
            case CLASS_ID_GOLD:
                scene_status.GoldScene = true;
                break;
            case CLASS_ID_CAR:
                scene_status.CarScene = true;
                break;
            case CLASS_ID_HUMAN:
                scene_status.HumanScene = true;
                break;
            case CLASS_ID_LIGHT_GREEN:
            case CLASS_ID_LIGHT_YELLOW:
            case CLASS_ID_LIGHT_RED:
                scene_status.LightScene = true;
                break;
            case CLASS_ID_SPEEDSIGN:
                scene_status.SpeedLimitScene = true;
                break;
            case CLASS_ID_BRANCHSIGN:
                scene_status.BranchScene = true;
                break;
            case CLASS_ID_STOP:
                scene_status.StopScene = true;
                break;
            case CLASS_ID_GO:
                scene_status.GoScene = true;
                break;
            default:
                break;
            }
        }
    }

    // 按 y2 从大到小排序
    std::sort(all_objects.begin(), all_objects.end(),
              [](const PredictResult &a, const PredictResult &b)
              { return a.y2 > b.y2; });

    // 清空上一帧数据，根据目标数预分配容量
    size_t n_cand = all_objects.size();
    gold_results_.clear();
    gold_results_.reserve(n_cand);
    car_results_.clear();
    car_results_.reserve(n_cand);
    human_results_.clear();
    human_results_.reserve(n_cand);
    light_results_.clear();
    light_results_.reserve(n_cand);
    zebraline_results_.clear();
    zebraline_results_.reserve(n_cand);
    speedlimit_results_.clear();
    speedlimit_results_.reserve(n_cand);
    branch_results_.clear();
    branch_results_.reserve(n_cand);
    stop_results_.clear();
    stop_results_.reserve(n_cand);
    go_results_.clear();
    go_results_.reserve(n_cand);

    gold_point.clear();
    gold_point.reserve(n_cand);
    car_point.clear();
    car_point.reserve(n_cand);
    human_point.clear();
    human_point.reserve(n_cand);
    light_point.clear();
    light_point.reserve(n_cand);
    zebra_point.clear();
    zebra_point.reserve(n_cand);
    speedlimit_point.clear();
    speedlimit_point.reserve(n_cand);
    branch_point.clear();
    branch_point.reserve(n_cand);
    stop_point.clear();
    stop_point.reserve(n_cand);
    go_point.clear();
    go_point.reserve(n_cand);

    t_gold_point.clear();
    t_gold_point.reserve(n_cand);
    t_car_point.clear();
    t_car_point.reserve(n_cand);
    t_human_point.clear();
    t_human_point.reserve(n_cand);
    t_light_point.clear();
    t_light_point.reserve(n_cand);
    t_zebra_point.clear();
    t_zebra_point.reserve(n_cand);
    t_speedlimit_point.clear();
    t_speedlimit_point.reserve(n_cand);
    t_branch_point.clear();
    t_branch_point.reserve(n_cand);
    t_stop_point.clear();
    t_stop_point.reserve(n_cand);
    t_go_point.clear();
    t_go_point.reserve(n_cand);

    // 写入对应全局数组（按 y2 降序：底部→顶部，由近到远）
    for (const auto &sel : all_objects)
    {
        int cid = sel.class_id;
        switch (cid)
        {
        case CLASS_ID_GOLD:
        {
            gold_results_.push_back(sel);
            int cx = (sel.x1 + sel.x2) / 2;
            int cy = sel.y2 + gold_fix_y;
            gold_point.push_back(cv::Point(cx, cy));
            cv::Point2f tp = transf(cx, cy);
            t_gold_point.push_back(cv::Point(tp.x, tp.y));
            break;
        }
        case CLASS_ID_CAR:
        {
            car_results_.push_back(sel);
            int cx = (sel.x1 + sel.x2) / 2;
            int cy = sel.y2;
            car_point.push_back(cv::Point(cx, cy));
            cv::Point2f tp = transf(cx, cy);
            t_car_point.push_back(cv::Point(tp.x, tp.y));
            break;
        }
        case CLASS_ID_HUMAN:
        {
            human_results_.push_back(sel);
            int cx = (sel.x1 + sel.x2) / 2;
            int cy = sel.y2;
            human_point.push_back(cv::Point(cx, cy));
            cv::Point2f tp = transf(cx, cy);
            t_human_point.push_back(cv::Point(tp.x, tp.y));
            break;
        }
        case CLASS_ID_LIGHT_GREEN:
        case CLASS_ID_LIGHT_YELLOW:
        case CLASS_ID_LIGHT_RED:
        {
            light_results_.push_back(sel);
            int cx = (sel.x1 + sel.x2) / 2;
            int cy = sel.y2;
            light_point.push_back(cv::Point(cx, cy));
            cv::Point2f tp = transf(cx, cy);
            t_light_point.push_back(cv::Point(tp.x, tp.y));
            break;
        }
        case CLASS_ID_ZEBRALINE:
        {
            zebraline_results_.push_back(sel);
            int cx = (sel.x1 + sel.x2) / 2;
            int cy = sel.y2;
            zebra_point.push_back(cv::Point(cx, cy));
            cv::Point2f tp = transf(cx, cy);
            t_zebra_point.push_back(cv::Point(tp.x, tp.y));
            break;
        }
        case CLASS_ID_SPEEDSIGN:
        {
            speedlimit_results_.push_back(sel);
            int cx = (sel.x1 + sel.x2) / 2;
            int cy = sel.y2;
            speedlimit_point.push_back(cv::Point(cx, cy));
            cv::Point2f tp = transf(cx, cy);
            t_speedlimit_point.push_back(cv::Point(tp.x, tp.y));
            break;
        }
        case CLASS_ID_BRANCHSIGN:
        {
            branch_results_.push_back(sel);
            int cx = (sel.x1 + sel.x2) / 2;
            int cy = sel.y2;
            branch_point.push_back(cv::Point(cx, cy));
            cv::Point2f tp = transf(cx, cy);
            t_branch_point.push_back(cv::Point(tp.x, tp.y));
            break;
        }
        case CLASS_ID_STOP:
        {
            stop_results_.push_back(sel);
            int cx = (sel.x1 + sel.x2) / 2;
            int cy = sel.y2;
            stop_point.push_back(cv::Point(cx, cy));
            cv::Point2f tp = transf(cx, cy);
            t_stop_point.push_back(cv::Point(tp.x, tp.y));
            break;
        }
        case CLASS_ID_GO:
        {
            go_results_.push_back(sel);
            int cx = (sel.x1 + sel.x2) / 2;
            int cy = sel.y2;
            go_point.push_back(cv::Point(cx, cy));
            cv::Point2f tp = transf(cx, cy);
            t_go_point.push_back(cv::Point(tp.x, tp.y));
            break;
        }
        default:
            break;
        }
    }
}

void Standard::trackRecognition(cv::Mat &frame, cv::Mat &mask, std::vector<cv::Point> &trackPoints_AI)
{
    trackPoints_AI.clear();
    filtered_line_AI.clear();
    t_trackPoints_AI.clear();
    s_t_trackPoints_AI.clear();
    d_s_t_trackPoints_AI.clear();
    t_CenterEdge.clear();

    if (trackstate == TrackState::TRACK_CV_MIDDLE)
    {
        trackPoints_AI = tracker.ExtractArrows_CV(frame, mask);
    }
    else if (trackstate == TrackState::TRACK_AI_MIDDLE)
    {
        trackPoints_AI = tracker.ExtractArrows_CV(frame, mask);
    }

    // 中线拟合
    filtered_line_AI = FitTrajectory_Poly((int)trackPoints_AI.size(), trackPoints_AI, frame);
}

// 曲率计算（弦高法）
double Standard::dynamicAimDisCal(const std::vector<cv::Point> &trackPoints_AI,
                                  std::vector<cv::Point2d> &double_pts)
{
    // ---- 先把所有点转成 double_pts（下游 computeError/processSpecials 依赖） ----
    double_pts.clear();
    double_pts.reserve(trackPoints_AI.size());
    for (const auto &pt : trackPoints_AI)
        double_pts.emplace_back(static_cast<double>(pt.x), static_cast<double>(pt.y));

    int total_sz = static_cast<int>(trackPoints_AI.size());
    if (total_sz < 10)
    {
        meanCurvature_ = 0.0;
        return 0.0;
    }

    // 1. 取预瞄视野内的起点和终点（避开最底部的车头盲区和最顶部的散点）
    int start_idx = static_cast<int>(total_sz * 0.1);
    int end_idx = static_cast<int>(total_sz * 0.9);

    cv::Point2f p_start(trackPoints_AI[start_idx].x, trackPoints_AI[start_idx].y);
    cv::Point2f p_end(trackPoints_AI[end_idx].x, trackPoints_AI[end_idx].y);

    // 2. 计算连接起点和终点的直线方程：Ax + By + C = 0
    float A = p_start.y - p_end.y;
    float B = p_end.x - p_start.x;
    float C = p_start.x * p_end.y - p_end.x * p_start.y;
    float denominator = std::hypot(A, B);

    double max_offset = 0.0;

    // 3. 遍历中间的所有点，计算它们到直线的垂直距离（寻找最大弦高）
    if (denominator > 1e-5f)
    {
        for (int i = start_idx + 1; i < end_idx; i++)
        {
            cv::Point2f pt(trackPoints_AI[i].x, trackPoints_AI[i].y);
            double dist = std::abs(A * pt.x + B * pt.y + C) / denominator;
            if (dist > max_offset)
                max_offset = dist;
        }
    }

    // 4. EMA 时序平滑（弦高法本身很稳定，alpha 可以稍大保证响应速度）
    const double ema_alpha = 0.4;
    meanCurvature_ = ema_alpha * max_offset + (1.0 - ema_alpha) * meanCurvature_;

    return meanCurvature_;
}