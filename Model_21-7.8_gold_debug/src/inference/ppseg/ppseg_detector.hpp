#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "../../../YOLO11_RK3588_object_detect-main/include/rknn_api.h"

// PpsegDetector: 基于 RKNN 的分割模型推理 C++ 封装。
// 使用说明：
//  - 调用 `load(model_path, img_size)` 加载并初始化 RKNN 上下文
//  - 调用 `infer(bgr_image, out_mask)` 执行推理并获取输出
// 输出说明（out_mask）：
//  - 类型：`cv::Mat`，数据类型为 `CV_8UC1`
//  - 尺寸：默认情况下会将模型输出的 mask 上采样到与传入 `infer` 的原始图像相同的宽高
//  - 像素值含义：
//      * 若模型输出为单通道概率图（或 logits），则 mask 以 0 表示背景、255 表示前景（阈值默认为 0.5，当数值范围在 [0,1] 时）
//      * 若模型输出为双通道（二分类的两个通道表示背景/前景），则 mask 以 0/255 表示背景/前景
//      * 若模型输出为多类分割（C>2 通道），则 mask 存储类别 id，取值在 [0, C-1]（以 8-bit 存储，超出 255 会被夹紧）
// 说明：
//  - 实现中会向 RKNN 请求浮点输出（float），并据此对输出进行解码得到 mask。
//  - 解码得到的 mask 空间分辨率与模型输出一致，随后根据需要使用最近邻插值放缩到原图尺寸。

class PpsegDetector
{
public:
    PpsegDetector();
    ~PpsegDetector();

    // Load model and initialize RKNN. img_size is a hint (not required if model input defines size).
    // npu_core: 0/1/2 for RK3588, set <0 to skip binding.
    bool load(const std::string &model_path, int img_size = 320, int npu_core = -1);
    void unload();
    bool is_loaded() const;

    // Run inference on a BGR image. out_mask is CV_8UC1 as described above.
    // If resize_to_input == true, the mask is scaled back to input image size using nearest neighbor.
    bool infer(const cv::Mat &bgr_image, cv::Mat &out_mask, bool resize_to_input = true);

    int model_width() const;
    int model_height() const;
    int model_channel() const;

private:
    struct ctx_t
    {
        rknn_context rknn_ctx;
        rknn_input_output_num io_num;
        rknn_tensor_attr *input_attrs;
        rknn_tensor_attr *output_attrs;
        int model_width;
        int model_height;
        int model_channel;
    } ctx_;

    bool loaded_;
    int input_size_; // bytes for uint8 input
    std::string model_path_;

    // decode a single RKNN output buffer (float) into a CV_8UC1 mask
    static bool decode_output_to_mask(const rknn_tensor_attr &attr, const float *buf, cv::Mat &mask);
};
