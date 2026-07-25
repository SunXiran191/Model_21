/**
 * thread/ocr_thread.cpp — Vision 语义导航工作线程
 * ============================================================
 * 架构: fire-and-forget 分离线程
 *   - 由 g_ocr_trigger (atomic bool) 触发
 *   - 从 G_produce 中获取 guide_road 检测框 & 原始帧
 *   - 裁剪路牌图片 → Base64 编码 → 上传视觉大模型直接推理
 *   - 结果写入 g_ocr_nav_result (atomic)
 *
 * 特点:
 *   - 线程独立于主控 run_thread，不阻塞飞控闭环
 *   - 图片直接发送给视觉LLM，无需本地OCR，消除误识别
 *   - 最多执行 ocr_max_tasks 次（岔路任务只两次）
 */

#include "thread.hpp"
#include "../inference/ocr/ocr_api.hpp"

#include <chrono>
#include <future>
#include <sched.h>
#include <string>
#include <vector>
#include <unistd.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

// 线程绑核辅助宏
#define PIN_TO_CORE(core)                                          \
    do                                                             \
    {                                                              \
        cpu_set_t cpu;                                             \
        CPU_ZERO(&cpu);                                            \
        CPU_SET(core, &cpu);                                       \
        pthread_setaffinity_np(pthread_self(), sizeof(cpu), &cpu); \
    } while (0)

std::atomic<int> g_ocr_nav_result{OCR_NAV_NONE};
std::atomic<int> g_ocr_nav_task_cnt{0};
std::atomic<int> g_ocr_nav_wait_frames{0};
std::atomic<bool> g_ocr_trigger{false};
std::string g_ocr_raw_text;

namespace
{
    static const int kOcrLoopUs = 5000;      // 轮询间隔 5ms
    static const int kVisionTimeoutSec = 30; // 视觉 API 超时 (秒)

    // ============================================================
    //  cv::Mat → Base64 (JPEG 编码)
    // ============================================================
    static std::string mat_to_base64(const cv::Mat &img, int quality = 70)
    {
        std::vector<uchar> buf;
        std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, quality};
        cv::imencode(".jpg", img, buf, params);

        static const char kTable[] =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        std::string result;
        result.reserve(((buf.size() + 2) / 3) * 4);
        for (size_t i = 0; i < buf.size(); i += 3)
        {
            uint32_t n = static_cast<uint32_t>(buf[i]) << 16;
            if (i + 1 < buf.size())
                n |= static_cast<uint32_t>(buf[i + 1]) << 8;
            if (i + 2 < buf.size())
                n |= static_cast<uint32_t>(buf[i + 2]);
            result += kTable[(n >> 18) & 0x3F];
            result += kTable[(n >> 12) & 0x3F];
            result += (i + 1 < buf.size()) ? kTable[(n >> 6) & 0x3F] : '=';
            result += (i + 2 < buf.size()) ? kTable[n & 0x3F] : '=';
        }
        return result;
    }
}

// ============================================================
//  infer_ocr_thread
// ============================================================
void infer_ocr_thread(Produce &G_produce, const Config &config)
{
    std::cout << "[Vision Thread] starting..." << std::endl;

    // 初始化百度视觉 API
    PIN_TO_CORE(1); // OCR 非实时，放小核 core 1
    BaiduApi api;

    while (!g_exit)
    {
        // 检查触发信号
        bool triggered = g_ocr_trigger.load(std::memory_order_acquire);
        if (!triggered)
        {
            usleep(kOcrLoopUs);
            continue;
        }

        // 检查执行次数限制
        int task_cnt = g_ocr_nav_task_cnt.load(std::memory_order_acquire);
        if (task_cnt >= config.ocr_max_tasks)
        {
            g_ocr_trigger.store(false, std::memory_order_release);
            usleep(kOcrLoopUs);
            continue;
        }

        // 获取guide_road检测框，裁剪
        const cv::Mat &frame = G_produce.img;
        const auto &preds = G_produce.predict_results;

        const PredictResult *best = nullptr;
        {
            extern std::mutex mtx_produce;
            std::lock_guard<std::mutex> lock(mtx_produce);

            if (frame.empty())
            {
                g_ocr_trigger.store(false, std::memory_order_release);
                usleep(kOcrLoopUs);
                continue;
            }

            for (const auto &obj : preds)
            {
                if (obj.class_id == CLASS_ID_BRANCHSIGN && obj.score >= 0.3f)
                {
                    if (!best || obj.score > best->score)
                        best = &obj;
                }
            }
        }

        if (!best)
        {
            g_ocr_trigger.store(false, std::memory_order_release);
            usleep(kOcrLoopUs);
            continue;
        }

        // 裁剪检测框
        int x1 = std::max(0, best->x1);
        int y1 = std::max(0, best->y1);
        int x2 = std::min(frame.cols - 1, best->x2);
        int y2 = std::min(frame.rows - 1, best->y2);

        if (x2 <= x1 || y2 <= y1)
        {
            g_ocr_trigger.store(false, std::memory_order_release);
            usleep(kOcrLoopUs);
            continue;
        }

        cv::Mat crop = frame(cv::Rect(x1, y1, x2 - x1, y2 - y1)).clone();

        std::cout << "[Vision Thread] task #" << task_cnt
                  << " bbox=(" << x1 << "," << y1 << "," << x2 << "," << y2 << ")"
                  << " crop=" << crop.cols << "x" << crop.rows << std::endl;

        // 执行视觉 API 管道（计时）
        g_ocr_nav_result.store(OCR_NAV_PENDING, std::memory_order_release);
        g_ocr_nav_wait_frames.store(0, std::memory_order_release);
        g_ocr_trigger.store(false, std::memory_order_release);
        g_ocr_nav_task_cnt.fetch_add(1, std::memory_order_release);

        auto t_start = std::chrono::steady_clock::now();

        // 1、图片 → Base64 编码
        std::string img_b64 = mat_to_base64(crop, 70);
        auto t_encode = std::chrono::steady_clock::now();
        std::cout << "[Vision Thread] base64: " << img_b64.size() << " bytes" << std::endl;

        // 2、上传视觉大模型直接推理（带超时保护）
        std::string decision;
        {
            auto future = std::async(std::launch::async, [&api, &img_b64]()
                                     { return api.navigate_vision(img_b64, "ernie-4.5-turbo-vl"); });
            if (future.wait_for(std::chrono::seconds(kVisionTimeoutSec)) == std::future_status::timeout)
            {
                std::cerr << "[Vision Thread] API timeout (" << kVisionTimeoutSec
                          << "s), fallback RIGHT" << std::endl;
                g_ocr_nav_result.store(OCR_NAV_RIGHT, std::memory_order_release);
                continue;
            }
            decision = future.get();
        }

        auto t_end = std::chrono::steady_clock::now();

        double encode_ms = std::chrono::duration<double, std::milli>(t_encode - t_start).count();
        double api_ms = std::chrono::duration<double, std::milli>(t_end - t_encode).count();
        double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        std::cout << "[Vision Thread] timing — encode: " << encode_ms
                  << " ms, API: " << api_ms << " ms, total: " << total_ms << " ms" << std::endl;

        if (decision.find("直行") != std::string::npos)
        {
            std::cout << "[Vision Thread] VLM → STRAIGHT" << std::endl;
            g_ocr_nav_result.store(OCR_NAV_STRAIGHT, std::memory_order_release);
        }
        else if (decision.find("右转") != std::string::npos)
        {
            std::cout << "[Vision Thread] VLM → RIGHT" << std::endl;
            g_ocr_nav_result.store(OCR_NAV_RIGHT, std::memory_order_release);
        }
        else
        {
            // "无法判断" 或其他 → 安全兜底右转
            std::cout << "[Vision Thread] VLM → UNCERTAIN, fallback RIGHT" << std::endl;
            g_ocr_nav_result.store(OCR_NAV_RIGHT, std::memory_order_release);
        }
    }

    std::cout << "[Vision Thread] stopped." << std::endl;
}
