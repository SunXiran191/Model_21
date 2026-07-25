#pragma once

#include <string>

// ============================================================
//  BaiduApi — 千帆 v2 OpenAI 兼容接口客户端
//  支持两种模式:
//    1. navigate(text) — 文字推理 (ernie-4.5-turbo-32k)
//    2. navigate_vision(image_base64) — 视觉推理 (看图直接判断)
//  输出: "直行" | "右转" | "无法判断"
//  环境变量:
//    BAIDU_API_KEY — IAM API Key
// ============================================================
class BaiduApi
{
public:
    BaiduApi();
    ~BaiduApi();

    void set_credentials(const std::string &api_key);

    /// @brief 文字语义导航
    std::string navigate(const std::string &ocr_text);

    /// @brief 视觉导航: 上传路牌图片直接推理 (星河API VL模型)
    /// @param image_base64  JPEG 编码的 base64 字符串
    /// @param model_name   模型名 (默认 ernie-4.5-turbo-vl)
    std::string navigate_vision(const std::string &image_base64,
                                const std::string &model_name = "ernie-4.5-turbo-vl");

    static std::string rule_based(const std::string &ocr_text);

private:
    std::string api_key_;
    bool enabled_;

    static std::string http_post(const std::string &url,
                                 const std::string &body,
                                 const std::string &bearer_token,
                                 int timeout_sec);
};
