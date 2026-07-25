#pragma once

#include <string>
#include <vector>

#include <netinet/in.h>

#include <opencv2/core.hpp>

#include "../../../res/include/aiget.hpp"
#include "ppyoloe_postprocess.hpp"

struct PpyoloeConfig
{
    // RKNN 模型文件路径，外部调用时必须提供有效 rknn 文件。
    std::string model_path = "modelInference_py/objDetect_1/model/rknn_lt.rknn";
    // 标签文件路径，按模型类别顺序逐行写入，例如 gold/car/human。
    std::string labels_path = "src/inference/ppyoloe/labels.txt";
    // 发送检测结果的 UDP 目标地址；启用后会把 JSON 结果发给 icar。
    std::string udp_ip = "127.0.0.1";
    int udp_port = 9000;
    // 是否在推理后自动发送 UDP JSON。
    bool enable_udp = true;
    // 低于该置信度的候选框会在后处理里被过滤。
    float confidence_threshold = 0.50f;
    // NMS 阈值，越小抑制越强。
    float nms_threshold = 0.45f;
    // 共享内存名称和映射大小，用于从摄像头/采集进程取实时图像。
    std::string shm_name = "shm_ar_video";
    size_t shm_map_bytes = 10 * 1024 * 1024;
    // NPU core binding: 0/1/2 for RK3588, set <0 to skip binding.
    int npu_core = 0;
};

// PPYOLOE 推理器封装。
// 用法：先 load(config)，再在循环里调用 infer(image, detections)，退出时调用 unload()。
class PpyoloeDetector
{
public:
    // 构造后对象不可直接推理，必须先 load()。
    PpyoloeDetector();
    ~PpyoloeDetector();

    // 加载 RKNN 模型、类别标签，并初始化可选的 UDP 发送目标。
    bool load(const PpyoloeConfig &config);
    // 释放模型、输出缓存和 UDP socket。
    void unload();
    // 对一帧 BGR 图像执行推理，输出坐标与类别结果。
    bool infer(const cv::Mat &bgr_image, std::vector<PredictResult> &detections);
    // 判断当前对象是否已经成功加载模型。
    bool is_loaded() const;

private:
    bool load_labels_();
    bool send_udp_json_(const std::vector<PredictResult> &detections) const;
    std::string build_udp_json_(const std::vector<PredictResult> &detections) const;

    ppyoloe_app_context_t ctx_;
    PpyoloeConfig config_;
    std::vector<std::string> labels_;
    bool loaded_;
    int udp_socket_;
    bool udp_ready_;
    sockaddr_in udp_addr_;

    // 预分配缓冲区，避免每帧 malloc/free
    rknn_tensor_mem *input_mem_;   // zero-copy non-cacheable input memory
    bool use_zero_copy_;           // 是否成功启用 zero-copy
    std::vector<unsigned char> input_buf_;  // fallback 输入缓冲区
    cv::Mat rgb_image_;
    cv::Mat resized_;
};