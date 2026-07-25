#include "ocr_detector.hpp"
#include "../../../YOLO11_RK3588_object_detect-main/include/rknn_api.h"
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <algorithm>

using namespace std;
using namespace cv;

// ============================================================
//  设置 NPU 核心绑定
// ============================================================
namespace
{
    static bool set_core_mask(rknn_context ctx, int core_id)
    {
        if (core_id < 0)
            return true;

        rknn_core_mask mask = RKNN_NPU_CORE_AUTO;
        switch (core_id)
        {
        case 0:
            mask = RKNN_NPU_CORE_0;
            break;
        case 1:
            mask = RKNN_NPU_CORE_1;
            break;
        case 2:
            mask = RKNN_NPU_CORE_2;
            break;
        default:
            return false;
        }
        return (rknn_set_core_mask(ctx, mask) == RKNN_SUCC);
    }
}

OcrDetector::OcrDetector()
    : loaded_(false), rknn_ctx_(0), io_num_(),
      input_attrs_(nullptr), output_attrs_(nullptr),
      model_channel_(0), model_width_(0), model_height_(0),
      input_fmt_(RKNN_TENSOR_NHWC), input_type_(RKNN_TENSOR_UINT8)
{
}

OcrDetector::~OcrDetector()
{
    release();
}

// ============================================================
//  加载模型 + 字典（使用 C API 直接加载 RKNN，无 Python 依赖）
// ============================================================
bool OcrDetector::load(const std::string &model_path, const std::string &dict_path)
{
    if (loaded_)
        release();

    // 1. 加载字符字典
    std::ifstream ifs(dict_path.c_str());
    if (!ifs.is_open())
    {
        std::cerr << "[OcrDetector] dict open failed: " << dict_path << std::endl;
        return false;
    }
    char_list_.clear();
    char_list_.push_back("blank"); // CTC blank (idx 0)
    std::string line;
    while (std::getline(ifs, line))
    {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        char_list_.push_back(line);
    }
    ifs.close();
    char_list_.push_back(" "); // space (idx 6624)
    std::cout << "[OcrDetector] dict loaded: " << char_list_.size() << " classes" << std::endl;

    // 2. 加载 RKNN 模型（C API，同进程 NPU 推理）
    std::ifstream model_ifs(model_path.c_str(), std::ios::binary);
    if (!model_ifs.is_open())
    {
        std::cerr << "[OcrDetector] model open failed: " << model_path << std::endl;
        return false;
    }
    std::vector<char> model_data((std::istreambuf_iterator<char>(model_ifs)),
                                 std::istreambuf_iterator<char>());
    model_ifs.close();
    if (model_data.empty())
    {
        std::cerr << "[OcrDetector] model data empty" << std::endl;
        return false;
    }

    rknn_context ctx = 0;
    int ret = rknn_init(&ctx, model_data.data(),
                        static_cast<uint32_t>(model_data.size()), 0, nullptr);
    if (ret < 0)
    {
        std::cerr << "[OcrDetector] rknn_init failed, ret=" << ret << std::endl;
        return false;
    }

    // 绑定到 NPU core 0（det=2, seg=1，core 0 空闲）
    if (!set_core_mask(ctx, 0))
        std::cerr << "[OcrDetector] WARNING: set_core_mask failed, using AUTO" << std::endl;

    // 查询输入/输出数量
    rknn_input_output_num io_num;
    memset(&io_num, 0, sizeof(io_num));
    if (rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num)) != RKNN_SUCC)
    {
        std::cerr << "[OcrDetector] query IO num failed" << std::endl;
        rknn_destroy(ctx);
        return false;
    }

    // 分配并查询输入/输出属性
    rknn_tensor_attr *in_attrs =
        static_cast<rknn_tensor_attr *>(malloc(io_num.n_input * sizeof(rknn_tensor_attr)));
    rknn_tensor_attr *out_attrs =
        static_cast<rknn_tensor_attr *>(malloc(io_num.n_output * sizeof(rknn_tensor_attr)));
    for (int i = 0; i < io_num.n_input; ++i)
    {
        memset(&in_attrs[i], 0, sizeof(rknn_tensor_attr));
        in_attrs[i].index = i;
        rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &in_attrs[i], sizeof(rknn_tensor_attr));
    }
    for (int i = 0; i < io_num.n_output; ++i)
    {
        memset(&out_attrs[i], 0, sizeof(rknn_tensor_attr));
        out_attrs[i].index = i;
        rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &out_attrs[i], sizeof(rknn_tensor_attr));
    }

    // 保存上下文
    rknn_ctx_ = ctx;
    io_num_ = io_num;
    input_attrs_ = in_attrs;
    output_attrs_ = out_attrs;

    // 解析输入格式（支持 NCHW 与 NHWC）
    input_fmt_ = in_attrs[0].fmt;
    input_type_ = in_attrs[0].type;
    if (input_fmt_ == RKNN_TENSOR_NCHW)
    {
        model_channel_ = in_attrs[0].dims[1];
        model_height_ = in_attrs[0].dims[2];
        model_width_ = in_attrs[0].dims[3];
    }
    else
    {
        model_height_ = in_attrs[0].dims[1];
        model_width_ = in_attrs[0].dims[2];
        model_channel_ = in_attrs[0].dims[3];
    }

    printf("[OcrDetector] model loaded: %dx%dx%d fmt=%d type=%d core=0\n",
           model_width_, model_height_, model_channel_,
           static_cast<int>(input_fmt_), static_cast<int>(input_type_));

    loaded_ = true;
    return true;
}

// ============================================================
//  OCR 识别（同进程 C++ RKNN 推理）
//  预处理: resize(h=48, ar) → pad(w=320) → BGR→RGB
//  推理:   rknn_inputs_set → rknn_run → rknn_outputs_get
//  解码:   CTC greedy decode
// ============================================================
std::string OcrDetector::recognize(const cv::Mat &crop)
{
    if (!loaded_ || crop.empty())
        return "";

    // ---- 预处理：resize(height=48) + pad(width=320) + BGR→RGB ----
    int h = crop.rows;
    int w = crop.cols;

    float ratio = 48.0f / static_cast<float>(h);
    int new_w = static_cast<int>(round(static_cast<float>(w) * ratio));
    if (new_w > model_width_)
        new_w = model_width_;

    cv::Mat resized;
    cv::resize(crop, resized, cv::Size(new_w, model_height_), 0, 0, cv::INTER_LINEAR);

    int pad_w = model_width_ - new_w;
    cv::Mat padded;
    if (pad_w > 0)
        cv::copyMakeBorder(resized, padded, 0, 0, 0, pad_w, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    else
        padded = resized;

    cv::Mat rgb;
    cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);

    // ---- 构造输入 buffer ----
    int num_elements = model_width_ * model_height_ * model_channel_;
    std::vector<unsigned char> input_buf(static_cast<size_t>(num_elements));

    if (input_fmt_ == RKNN_TENSOR_NCHW)
    {
        // HWC → CHW 转置
        for (int c = 0; c < model_channel_; ++c)
            for (int y = 0; y < model_height_; ++y)
            {
                const uchar *row = rgb.ptr<uchar>(y);
                for (int x = 0; x < model_width_; ++x)
                    input_buf[c * model_height_ * model_width_ + y * model_width_ + x] =
                        row[x * 3 + c];
            }
    }
    else
    {
        // NHWC 直接拷贝
        memcpy(input_buf.data(), rgb.data, static_cast<size_t>(num_elements));
    }

    // ---- 设定输入 ----
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = input_fmt_;
    inputs[0].size = static_cast<uint32_t>(num_elements);
    inputs[0].buf = input_buf.data();
    if (rknn_inputs_set(rknn_ctx_, 1, inputs) < 0)
    {
        std::cerr << "[OcrDetector] rknn_inputs_set failed" << std::endl;
        return "";
    }

    // ---- 推理 ----
    if (rknn_run(rknn_ctx_, nullptr) < 0)
    {
        std::cerr << "[OcrDetector] rknn_run failed" << std::endl;
        return "";
    }

    // ---- 获取输出 ----
    std::vector<rknn_output> outputs(io_num_.n_output);
    memset(outputs.data(), 0, io_num_.n_output * sizeof(rknn_output));
    for (int i = 0; i < io_num_.n_output; ++i)
    {
        outputs[i].index = i;
        outputs[i].want_float = 1;
    }
    if (rknn_outputs_get(rknn_ctx_, io_num_.n_output, outputs.data(), nullptr) < 0)
    {
        std::cerr << "[OcrDetector] rknn_outputs_get failed" << std::endl;
        return "";
    }

    // ---- CTC 贪心解码 ----
    // output[0]: (1, T, num_classes) float32
    // T = n_elems / char_list_.size()
    int C = static_cast<int>(char_list_.size());
    uint32_t total_elems = output_attrs_[0].n_elems;
    int T = static_cast<int>(total_elems) / C;

    float *probs = static_cast<float *>(outputs[0].buf);
    std::string result = ctc_decode(probs, T, C, char_list_);

    // ---- 释放输出 ----
    rknn_outputs_release(rknn_ctx_, io_num_.n_output, outputs.data());

    std::cout << "[OcrDetector] OCR result: '" << result << "'" << std::endl;
    return result;
}

void OcrDetector::release()
{
    if (input_attrs_)
    {
        free(input_attrs_);
        input_attrs_ = nullptr;
    }
    if (output_attrs_)
    {
        free(output_attrs_);
        output_attrs_ = nullptr;
    }
    if (rknn_ctx_ != 0)
    {
        rknn_destroy(rknn_ctx_);
        rknn_ctx_ = 0;
    }
    loaded_ = false;
    char_list_.clear();
}

// ============================================================
//  CTC 贪心解码 (静态)
// ============================================================
std::string OcrDetector::ctc_decode(const float *probs, int T, int C,
                                    const std::vector<std::string> &char_list)
{
    std::string result;
    int prev = 0;
    for (int t = 0; t < T; ++t)
    {
        int best = 0;
        float best_val = probs[t * C];
        for (int c = 1; c < C; ++c)
        {
            if (probs[t * C + c] > best_val)
            {
                best_val = probs[t * C + c];
                best = c;
            }
        }
        if (best == 0)
        {
            prev = 0;
            continue;
        }
        if (best == prev)
            continue;
        if (best < (int)char_list.size())
            result += char_list[best];
        prev = best;
    }
    return result;
}
