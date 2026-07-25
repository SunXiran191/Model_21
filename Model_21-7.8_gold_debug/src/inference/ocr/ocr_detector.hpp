#pragma once

#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include "../../../YOLO11_RK3588_object_detect-main/include/rknn_api.h"

// ============================================================
//  OcrDetector — PP-OCRv4 识别模型封装（C++ RKNN 直接推理）
//  与主进程共享 NPU，避免子进程资源冲突
//  输入: BGR 裁剪文本图像
//  输出: 识别文字字符串
// ============================================================
class OcrDetector
{
public:
    OcrDetector();
    ~OcrDetector();

    /// @brief 加载 RKNN 模型与字符字典（使用 C API 直接加载）
    /// @param model_path  .rknn 文件路径
    /// @param dict_path   字符字典文件路径 (ppocr_keys_v1.txt)
    /// @return true 成功
    bool load(const std::string &model_path, const std::string &dict_path);

    /// @brief 是否已加载
    bool is_loaded() const { return loaded_; }

    /// @brief 对裁剪图像执行 OCR 识别（同进程 NPU 推理）
    /// @param crop  BGR 格式文本区域图像 (矩形裁剪)
    /// @return 识别出的文字，失败返回空字符串
    std::string recognize(const cv::Mat &crop);

    /// @brief 释放 NPU 资源
    void release();

private:
    bool loaded_;
    rknn_context rknn_ctx_;
    rknn_input_output_num io_num_;
    rknn_tensor_attr *input_attrs_;
    rknn_tensor_attr *output_attrs_;
    int model_channel_;
    int model_width_;
    int model_height_;
    rknn_tensor_format input_fmt_;
    rknn_tensor_type input_type_;
    std::vector<std::string> char_list_; // 6625 类字符字典

    // 内部: CTC 贪心解码
    static std::string ctc_decode(const float *probs, int T, int C,
                                  const std::vector<std::string> &char_list);
};
