#pragma once

#include <cstddef>
#include <string>
#include "../include/json.hpp"

struct Config
{
    // ！！！！！！！！！！以下所有具体数值在config中改！！！！！！！！！！！！
    std::string jsonpath = "$.config";
    std::string model_path = "./model";

    std::string infer_model_path = "modelInference_py/objDetect_1/model/rknn_lt.rknn";
    std::string infer_labels_path = "src/inference/ppyoloe/labels.txt";
    std::string infer_seg_model_path = "modelInference_py/segmentation/model/pp_liteseg320.rknn";
    // std::string infer_udp_ip = "127.0.0.1";

    std::string ocr_model_path = "modelInference_py/model/ppocr_v4_rec_rk3588.rknn";
    std::string ocr_dict_path = "objDetect_1/ppocr_keys_v1.txt";
    int ocr_max_tasks = 2; // 岔路 OCR 最多执行次数

    int infer_udp_port = 9000;
    bool infer_enable_udp = true;
    float infer_nms_threshold = 0.45f;
    // std::string infer_shm_name = "shm_ar_video";
    // size_t infer_shm_map_bytes = 10 * 1024 * 1024;
    //  NPU core binding: 0/1/2 for RK3588, set <0 to skip binding.

    std::string save_image_dir = "./saved_images";

    float speedHigh = 1.17f;

    float aim_distance_f = 0.8f;        // aim_distance_f*PIXPERMETER = 预瞄点的像素距离。现在PIXPERMETER = 100，预瞄点在逆透视后中线距底部中点200pix距离
    float aim_distance_branch_f = 0.8f; // 岔路预瞄点距离
    float confidence_threshold = 0.3f;  // 目标检测置信度

    int car_avoid_dist = 120; // 车避障：绕行偏移距离（IPM 像素）

    int human_near_threshold = 100;     // 行人近/远区分界阈值（IPM像素），|偏移| < 此值为近区
    int human_control_y_threshold = 50; // 只有行人点 Y 坐标大于该阈值时才进入控制状态判断
    int human_plus_near_dist = 20;
    int human_close_avoid_dist = 120; // 行人近距离避障距离（IPM像素）
    int human_stop_threshold = 100;

    int human_speed_smooth = 0.5f; // 行人速度 EMA 平滑系数 (0~1，越大越平滑)

    int CAR_FIND_FRAMES_THRESH = 2;   // 车辆：连续检测多少帧才确认发现
    int CAR_LOST_FRAMES_THRESH = 5;   // 车辆：连续丢失多少帧才判定丢失
    int GOLD_FIND_FRAMES_THRESH = 2;  // 金币：连续检测多少帧才确认发现
    int GOLD_LOST_FRAMES_THRESH = 5;  // 金币：连续丢失多少帧才判定丢失
    int HUMAN_FIND_FRAMES_THRESH = 2; // 行人：连续检测多少帧才确认发现
    int HUMAN_LOST_FRAMES_THRESH = 5; // 行人：连续丢失多少帧才判定丢失

    int GOLD_DIST_FORWARD = 40;   // 金币屏蔽区：向前延伸像素数（屏蔽该区域内的原巡线点）
    int GOLD_DIST_BACKWARD = 40;  // 金币屏蔽区：向后延伸像素数
    int CAR_DIST_FORWARD = 40;    // 车辆屏蔽区：向前延伸像素数
    int CAR_DIST_BACKWARD = 50;   // 车辆屏蔽区：向后延伸像素数
    int HUMAN_DIST_FORWARD = 40;  // 行人屏蔽区：向前延伸像素数
    int HUMAN_DIST_BACKWARD = 50; // 行人屏蔽区：向后延伸像素数
    int max_gold_track_dist = 50; // 金币距离赛道中心最大允许距离像素，超过则放弃该金币
    int gold_fix_y = 10;

    int go_stop_dist_forward = 80;  // STOP/GO 标志牌：沿拟合斜率向上（图像上方）延长距离（IPM像素）
    int go_stop_dist_backward = 40; // STOP/GO 标志牌：沿拟合斜率向下（图像下方）延长距离（IPM像素）

    int dense_y_low = 50;     // 密集目标检测Y范围下限（IPM像素）
    int dense_y_high = 400;   // 密集目标检测Y范围上限（IPM像素）
    float dense_speed = 0.7f; // 密集目标减速目标速度（m/s）

    int track_width_left = 0;    // 左锚点 ROI 宽度（主路拟合范围，0=左边缘）
    int track_width_right = 300; // 右锚点 ROI 宽度（岔路拟合范围）

    int branch_detect_frame_threshold = 12; // NONE→DETECTED: 岔路标志累计帧数阈值
    int branch_in_frame_threshold = 20;     // 岔路标志丢失→进入岔路帧数阈值
    int branch_out_frame_threshold = 2;     // 检测到T字路口→BRANCH_OUT帧数阈值
    int branch_over_frame_threshold = 20;   // T字路口消失→BRANCH_OVER帧数阈值
    int branch_none_frame_threshold = 28;   // 回NONE帧数阈值
    int branch_ocr_speed = 1.0f;
    int branch_sign_width_threshold = 62;   // 岔路标志检测框最小宽度（像素）
    int branch_sign_edge_x_margin = 30;     // 标志框各边距画面边缘的最小距离（像素）
    int branch_sign_edge_y_margin = 30;     // 标志框各边距画面边缘的最小距离（像素）
    float branch_sign_area_ratio_x2 = 2.5f; // 进入 AWAIT：当前面积/初始面积 阈值

    float t_junction_margin = 20.0f;  // T字路口检测：边缘外推距离（IPM 像素）
    int t_junction_row_threshold = 8; // T字路口检测：超出基线行数阈值

    float light_stop_distance_m = 0.18f; // 停车时距斑马线距离

    float fuzzy_C_STOP = 0.00f; // 模糊输出：STOP 中心值
    float fuzzy_C_SLOW = 0.10f; // 模糊输出：SLOW 中心值
    float fuzzy_C_KEEP = 0.85f; // 模糊输出：KEEP 中心值
    float fuzzy_C_FULL = 1.00f; // 模糊输出：FULL 中心值

    // 调试输出控制（debug=true 开启全部；各分项独立控制）
    bool debug = false;
    bool infer_det_debug = false;
    bool run_debug = false;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Config,
                                   jsonpath, model_path,
                                   infer_model_path, infer_labels_path, infer_seg_model_path,
                                   infer_udp_port,
                                   infer_enable_udp,
                                   infer_nms_threshold,
                                   save_image_dir,
                                   speedHigh,
                                   aim_distance_f, aim_distance_branch_f,
                                   confidence_threshold,
                                   car_avoid_dist, human_near_threshold, human_plus_near_dist,
                                   human_control_y_threshold, human_close_avoid_dist,
                                   CAR_FIND_FRAMES_THRESH,
                                   CAR_LOST_FRAMES_THRESH,
                                   GOLD_FIND_FRAMES_THRESH, GOLD_LOST_FRAMES_THRESH,
                                   HUMAN_FIND_FRAMES_THRESH, HUMAN_LOST_FRAMES_THRESH,
                                   GOLD_DIST_FORWARD,
                                   GOLD_DIST_BACKWARD,
                                   CAR_DIST_FORWARD, CAR_DIST_BACKWARD,
                                   HUMAN_DIST_FORWARD, HUMAN_DIST_BACKWARD,
                                   max_gold_track_dist, gold_fix_y,
                                   go_stop_dist_forward, go_stop_dist_backward,
                                   dense_y_low, dense_y_high, dense_speed,
                                   track_width_left, track_width_right,
                                   branch_detect_frame_threshold, branch_in_frame_threshold,
                                   branch_out_frame_threshold, branch_over_frame_threshold,
                                   branch_none_frame_threshold,
                                   t_junction_margin, t_junction_row_threshold,
                                   branch_sign_width_threshold,
                                   branch_sign_edge_x_margin, branch_sign_edge_y_margin,
                                   branch_sign_area_ratio_x2,
                                   ocr_model_path, ocr_dict_path, ocr_max_tasks,
                                   light_stop_distance_m,
                                   fuzzy_C_STOP, fuzzy_C_SLOW, fuzzy_C_KEEP, fuzzy_C_FULL,
                                   debug, infer_det_debug, run_debug);
};
