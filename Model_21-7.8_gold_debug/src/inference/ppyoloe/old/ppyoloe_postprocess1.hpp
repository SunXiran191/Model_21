#pragma once

#include <vector>

#include "../../../res/include/aiget.hpp"

#include "../../../YOLO11_RK3588_object_detect-main/include/rknn_api.h"

struct ppyoloe_app_context_t
{
    rknn_context rknn_ctx = 0;
    rknn_input_output_num io_num{};
    rknn_tensor_attr *input_attrs = nullptr;
    rknn_tensor_attr *output_attrs = nullptr;
    int model_channel = 0;
    int model_width = 0;
    int model_height = 0;
    bool is_quant = false;

    // 后处理缓存：避免每帧重复检测
    bool need_sigmoid = false;       // 模型输出是否需要手动 sigmoid
    bool sigmoid_detected = false;   // 是否已完成首次检测
    rknn_tensor_format input_fmt = RKNN_TENSOR_NHWC;  // 模型实际输入格式
    rknn_tensor_type input_type = RKNN_TENSOR_UINT8;   // 模型实际输入类型
};

// 初始化后处理所需资源，当前实现主要用于接口统一，外部调用可在程序启动时执行一次。
int ppyoloe_init_post_process();
// 释放后处理侧预留资源，退出时调用即可。
void ppyoloe_deinit_post_process();
// 根据类别 id 返回类别名，适合调试、日志或绘制标签。
char *ppyoloe_cls_to_name(int cls_id);
// 将 RKNN 原始输出解码为 icar 侧使用的 PredictResult 列表。
// 调用前应保证 app_ctx 已加载模型，outputs 来自 rknn_outputs_get。
int ppyoloe_post_process(ppyoloe_app_context_t *app_ctx,
                         void *outputs,
                         int img_width,
                         int img_height,
                         float conf_threshold,
                         float nms_threshold,
                         std::vector<PredictResult> &detections);