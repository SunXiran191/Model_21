#pragma once

#include <vector>
#include <cmath>
#include <iostream>
#include <chrono>
#include <opencv2/core.hpp>

// #include "../../res/include/common.hpp"
#include "general.hpp"
#include "../common/utils.hpp"
#include "../special/go_stop.hpp"
#include "../special/light.hpp"
#include "../special/objects.hpp"
#include "../special/human.hpp"
#include "../special/branch.hpp"
#include "../imgprocess/LineTracker.hpp"
#include "../imgprocess/transform.hpp"
#include "../imgprocess/imgProcess.hpp"
#include "../../res/configs/param.hpp"
#include "../../res/include/aiget.hpp"
#include "fuzzy.hpp"

using namespace std;

struct Produce;

// 实时数据，任务逻辑输出数据
struct TaskData
{
    cv::Mat img;
    // std::chrono::time_point<std::chrono::_v2::steady_clock, std::chrono::duration<long int, std::ratio<1, 1000000000>>> timestamp;
    bool beep_enable = 0;
    float speed = 1.0f;
    float error = 0.0f;
    float x_error = 0.0f;
    bool bend_flag = false; // 弯道标志
    bool lost_flag = false; // 停止标志
    int track_side = 0;     // 丢线方向: -1=车在赛道左侧, 0=未知/未丢线, 1=车在赛道右侧（仅lost_cnt>3时有效）
    std::vector<PredictResult> predict_results;
};

// 曲率计算中间结构
struct PointsCurve
{
    double x, y;
    double dx, dy;
    double ddx, ddy;
    double curve;
};

class Scene_status
{
public:
    bool GoldScene;
    bool CarScene;
    bool HumanScene;
    bool LightScene;
    bool BranchScene;
    bool SpeedLimitScene;
    bool StopScene;
    bool GoScene;

    Scene_status();
    bool all();
    // bool none();
};

class Standard
{
public:
    Standard();
    Standard(Config config);

    enum TrackState
    {
        TRACK_CV_MIDDLE = 0,
        TRACK_AI_MIDDLE = 1
    };
    TrackState trackstate = TRACK_AI_MIDDLE;

    General general;
    Config config_;
    Scene_status scene_status;
    imgProcess img_process;

    LineTracker tracker;
    Objects objects_;
    Human human_;
    Light light_;
    GoStop go_stop_;
    FuzzyHumanSpeedController fuzzy_human_speed_; // 行人速度模糊控制器

    enum Scene_status_e
    {
        Normal_status = 0,
        stop_status,
        branch_status,
        speedlimit_status,
        Light_status,
    };
    Scene_status_e scene_status_e = Normal_status;

    float aim_distance_f;
    int gold_fix_y = 0;
    float pre_error;
    int human_stop_threshold = 0;
    float gold_in_angle = 0.0f;

    vector<PredictResult> gold_results_;
    vector<PredictResult> car_results_;
    vector<PredictResult> human_results_;
    vector<PredictResult> light_results_;
    vector<PredictResult> zebraline_results_;
    vector<PredictResult> speedlimit_results_;
    vector<PredictResult> branch_results_;
    vector<PredictResult> stop_results_;
    vector<PredictResult> go_results_;

    vector<cv::Point> gold_point;
    vector<cv::Point> car_point;
    vector<cv::Point> human_point;
    vector<cv::Point> light_point;
    vector<cv::Point> zebra_point;
    vector<cv::Point> speedlimit_point;
    vector<cv::Point> branch_point;
    vector<cv::Point> stop_point;
    vector<cv::Point> go_point;

    vector<cv::Point> t_gold_point;
    vector<cv::Point> t_car_point;
    vector<cv::Point> t_human_point;
    vector<cv::Point> t_light_point;
    vector<cv::Point> t_zebra_point;
    vector<cv::Point> t_speedlimit_point;
    vector<cv::Point> t_branch_point;
    vector<cv::Point> t_stop_point;
    vector<cv::Point> t_go_point;

    cv::Point2f start_s_t_trackPoints_CV;

    vector<cv::Point> trackPoints_CV;
    vector<cv::Point> filtered_line_CV;     // 传统cv赛道拟合后点集
    vector<cv::Point> t_trackPoints_CV;     // 透视变换后传统cv赛道点集
    vector<cv::Point> s_t_trackPoints_CV;   // 等距采样后传统cv赛道点集
    vector<cv::Point> d_s_t_trackPoints_CV; // 元素拟合后传统cv赛道点集

    cv::Point aim_point; // 当前预瞄点（用于可视化）

    vector<cv::Point> trackPoints_AI;       // AI分割赛道点集
    vector<cv::Point> filtered_line_AI;     // 传统cv赛道拟合后点集
    vector<cv::Point> t_trackPoints_AI;     // 透视变换后AI分割赛道点集
    vector<cv::Point> s_t_trackPoints_AI;   // 等距采样后AI分割赛道点集
    vector<cv::Point> d_s_t_trackPoints_AI; // 元素拟合后AI分割赛道点集
    vector<cv::Point> t_CenterEdge;         // 曲率计算

    TaskData run(Produce &G_produce, float &run_speed, double run_fps = 0.0f);
    void updateSceneStatus(const std::vector<PredictResult> &frame, Scene_status &scene_status, const Config *config);
    double dynamicAimDisCal(const std::vector<cv::Point> &trackPoints_AI, std::vector<cv::Point2d> &double_pts);

    // 丢线方向
    float side_history_[3] = {0.0f, 0.0f, 0.0f};
    int side_history_idx_ = 0;
    int side_history_cnt_ = 0;

    void trackRecognition(cv::Mat &frame, cv::Mat &mask, std::vector<cv::Point> &trackPoints);

private:
    // 从Produce拷贝帧数据（带锁）
    bool extractFrameData(Produce &G_produce, cv::Mat &src_img, cv::Mat &src_segmask,
                          std::vector<PredictResult> &predict_result);

    // 巡线提取
    void extractLanePoints(const cv::Mat &src_img, const cv::Mat &src_segmask,
                           const std::vector<PredictResult> &predict_result,
                           cv::Mat &morph_segmask, cv::Mat &mask, Produce &G_produce,
                           const float speed);

    // 等距采样
    bool computeTrajectory(float &cx, float &cy, cv::Point2f &car_base_ipm, int &traj_sz);

    // 特殊元素处理：灯光/障碍物/金币/行人/停车
    void processSpecials(const std::vector<PredictResult> &predict_result, cv::Mat &src_img,
                         int &traj_sz, float &run_speed, double run_fps, TaskData &data,
                         const cv::Point2f &car_base_ipm);

    // 偏差计算 + 丢线恢复
    void computeError(float cx, float cy, const cv::Point2f &car_base_ipm,
                      int traj_sz, TaskData &data);

    // 速度决策 + 密集目标减速
    void speedDecision(float &ctrl_speed, TaskData &data, cv::Point2f &car_base_ipm);

    // 性能统计输出（每秒一次）
    void printStats(double copy_ms, double update_ms, double extract_ms,
                    double fit_ms, double curve_ms, double objects_ms, double error_ms);

    double fit_ms_ = 0.0; // 本帧拟合耗时

    // ====== 每帧运行状态 ======
    int lost_cnt_ = 0;               // 连续丢线帧数
    int mask_area_threshold_ = 2000; // mask 面积阈值：低于此值时认为赛道丢失，递增丢线计数
    int stop_delay_cnt_ = -1;        // 停车倒计时计数
    int err_log_cnt_ = 0;
    int str_speed_up_cnt_ = 0;                                        // 强制加速计数
    uint64_t last_seg_seq_ = 0;                                       // 上次读取的 seg_mask
    float human_offset_ = 0.0f;                                       // 最近行人偏移量
    int pedestrian_recovery_cnt_ = 0;                                 // 行人避让结束后加速恢复的剩余帧数
    Human::ControlState prev_human_ctrl_ = Human::ControlState::NONE; // 上一帧行人控制状态，用于检测切换

    // ====== 性能统计状态 ======
    uint64_t prof_frames_ = 0;
    double prof_copy_ = 0.0, prof_update_ = 0.0;
    double prof_extract_ = 0.0;
    double prof_fit_ = 0.0;
    double prof_curve_ = 0.0;
    double prof_objects_ = 0.0, prof_error_ = 0.0;
    std::chrono::steady_clock::time_point prof_last_ = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point last_frame_time_;

    float cx = IMAGE_W / 2.0f; // 车轮对应点 (纯跟踪起始点)
    float cy = IMAGE_H * 0.95f;
    int aim_index_far = 0;
    int aim_index_near = 0;
    double wheelbase = 0.20;     // 车身长
    double meanCurvature_ = 0.0; // 上一帧中线的弦高（像素），越大弯越急

    float actual_fps_ = 30.0f;

    float aim_distance_n;
    float aim_angle_p_k;
    float aim_angle_p;
    float aim_angle_d;

    float max_speed_ = 0.0f;             // 当前运行的目标上限速度
    float human_distance_thresh_ = 0.0f; // 行人距离阈值

    float dist_human_reach = 0.0f; // 行人与车所在点的直线距离

public:
    float aim_speed = 0.f;
    float aim_angle = 0.0f;      // 偏差量
    float aim_angle_last = 0.0f; // 偏差量 上一帧
    float aim_sigma = 0.0f;      // 偏差方差
};
