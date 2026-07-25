#pragma once

#include <thread>
#include <mutex>
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <deque>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <signal.h>
#include "../standard/standard.hpp"
#include "../../res/include/video_get.hpp"
#include "../../res/include/mjpeg_streamer.hpp"
#include "../imgprocess/LineTracker.hpp"
#include "../common/utils.hpp"
#include "../standard/general.hpp"
#include "../../res/include/sendUART.hpp"
// #include "../../res/include/aiget.hpp"
#include "../imgprocess/transform.hpp"
#include "../../res/configs/param.hpp"
#include "../inference/ppyoloe/ppyoloe_detector.hpp"
#include "../../res/include/control.hpp"

extern atomic<bool> g_exit;
extern atomic<bool> stop_flag;

extern mutex mtx_produce;
extern atomic<uint64_t> g_produce_seq;
extern atomic<uint64_t> g_render_seq;
extern atomic<uint64_t> g_seg_seq;   // seg_mask 版本号，由 infer_seg_thread 递增

extern cv::Point g_aim_point;

// extern Uart uart;
extern atomic<double> g_run_thread_fps;
extern atomic<float> speed_now;
extern atomic<float> g_error;
extern void resample_points(const std::vector<cv::Point> &input, int input_size, std::vector<cv::Point> &output, int &output_size, float dist_threshold);

// ============================================================
//  OCR 语义导航 全局状态
//  由 infer_ocr_thread 写入，Branch 状态机读取
// ============================================================
enum OcrNavResult
{
    OCR_NAV_NONE = 0,     // 空闲
    OCR_NAV_PENDING,      // OCR+API 运行中
    OCR_NAV_STRAIGHT,     // 结果: 直行
    OCR_NAV_RIGHT,        // 结果: 右转
};
extern std::atomic<int> g_ocr_nav_result;   // OcrNavResult
extern std::atomic<int> g_ocr_nav_task_cnt; // 已执行次数 (上限2)
extern std::atomic<int> g_ocr_nav_wait_frames; // 等待帧计数

// 由 branch 提供给 OCR 线程: 是否需要处理 (BRANCH_DETECTED 时为 true)
extern std::atomic<bool> g_ocr_trigger;
// OCR 识别到的原始文字 (调试用)
extern std::string g_ocr_raw_text;

struct Produce
{
	cv::Mat img;
	// PredictFrame predict_frame;
	cv::Mat seg_mask;
	cv::Mat morph_mask;
	cv::Mat ipm_mask; // debug: IPM ��͸�Ӻ��������ӻ�
	std::vector<PredictResult> predict_results;
	// ===== LATENCY_MEASURE =====
	uint64_t frame_seq = 0;  // 当前帧对应的 g_frame_seq 编号
	// ===== LATENCY_MEASURE_END =====
};

void drawBox(cv::Mat &img, const std::vector<PredictResult> &predict_results);
void produce_thread(Produce &G_produce, const Config &config);
void infer_det_thread(Produce &G_produce, const Config &config);
void infer_seg_thread(Produce &G_produce, const Config &config);
void infer_ocr_thread(Produce &G_produce, const Config &config);  // OCR 语义导航线程
void run_thread(Produce &G_produce, const Config &config);
void stream_thread(Produce &G_produce);
void capture_thread(Produce &G_produce);
