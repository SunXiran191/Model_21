#include "ppyoloe_detector.hpp"

#include <arpa/inet.h>
#include <chrono>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <cstring>

#include <opencv2/imgproc.hpp>

namespace
{
    static std::string read_text_file(const std::string &path)
    {
        std::ifstream ifs(path.c_str());
        if (!ifs.is_open())
            return std::string();
        std::string text, line;
        while (std::getline(ifs, line))
        {
            if (!line.empty() && line[line.size() - 1] == '\r')
                line.erase(line.size() - 1);
            if (!text.empty())
                text.push_back('\n');
            text += line;
        }
        return text;
    }

    static bool set_core_mask_if_needed(rknn_context ctx, int core_id, const char *tag)
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

    static std::string json_escape(const std::string &in)
    {
        std::string out;
        out.reserve(in.size());
        for (char c : in)
        {
            if (c == '"' || c == '\\')
                out.push_back('\\');
            out.push_back(c);
        }
        return out;
    }
}

PpyoloeDetector::PpyoloeDetector() : loaded_(false), udp_socket_(-1), udp_ready_(false)
{
    ctx_ = ppyoloe_app_context_t{};
    memset(&udp_addr_, 0, sizeof(udp_addr_));
}

PpyoloeDetector::~PpyoloeDetector() { unload(); }

bool PpyoloeDetector::load_labels_()
{
    labels_.clear();
    const std::string text = read_text_file(config_.labels_path);
    if (text.empty())
        return false;
    std::istringstream iss(text);
    std::string line;
    while (std::getline(iss, line))
    {
        if (!line.empty())
            labels_.push_back(line);
    }
    return !labels_.empty();
}

bool PpyoloeDetector::load(const PpyoloeConfig &config)
{
    unload();
    config_ = config;
    if (!load_labels_())
        return false;

    std::ifstream ifs(config_.model_path.c_str(), std::ios::binary);
    if (!ifs.is_open())
        return false;
    std::vector<char> model_data((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    if (model_data.empty())
        return false;

    rknn_context ctx = 0;
    if (rknn_init(&ctx, model_data.data(), static_cast<uint32_t>(model_data.size()), 0, NULL) < 0)
        return false;
    set_core_mask_if_needed(ctx, config_.npu_core, "PPYOLOE");

    rknn_input_output_num io_num{};
    if (rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num)) != RKNN_SUCC)
        return false;

    std::vector<rknn_tensor_attr> input_attrs(io_num.n_input);
    std::vector<rknn_tensor_attr> output_attrs(io_num.n_output);
    for (int i = 0; i < io_num.n_input; ++i)
    {
        memset(&input_attrs[i], 0, sizeof(rknn_tensor_attr));
        input_attrs[i].index = i;
        rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &input_attrs[i], sizeof(rknn_tensor_attr));
    }
    for (int i = 0; i < io_num.n_output; ++i)
    {
        memset(&output_attrs[i], 0, sizeof(rknn_tensor_attr));
        output_attrs[i].index = i;
        rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &output_attrs[i], sizeof(rknn_tensor_attr));
    }

    ctx_.rknn_ctx = ctx;
    ctx_.io_num = io_num;
    ctx_.input_attrs = static_cast<rknn_tensor_attr *>(malloc(io_num.n_input * sizeof(rknn_tensor_attr)));
    ctx_.output_attrs = static_cast<rknn_tensor_attr *>(malloc(io_num.n_output * sizeof(rknn_tensor_attr)));
    memcpy(ctx_.input_attrs, input_attrs.data(), io_num.n_input * sizeof(rknn_tensor_attr));
    memcpy(ctx_.output_attrs, output_attrs.data(), io_num.n_output * sizeof(rknn_tensor_attr));

    ctx_.model_channel = input_attrs[0].dims[1];
    ctx_.model_height = input_attrs[0].dims[2];
    ctx_.model_width = input_attrs[0].dims[3];
    if (input_attrs[0].fmt == RKNN_TENSOR_NHWC)
    {
        ctx_.model_height = input_attrs[0].dims[1];
        ctx_.model_width = input_attrs[0].dims[2];
        ctx_.model_channel = input_attrs[0].dims[3];
    }

    ctx_.input_fmt = input_attrs[0].fmt;
    ctx_.input_type = input_attrs[0].type;

    printf("[PPYOLOE] Model loaded: %dx%dx%d\n", ctx_.model_width, ctx_.model_height, ctx_.model_channel);
    loaded_ = true;
    return true;
}

void PpyoloeDetector::unload()
{
    if (ctx_.input_attrs != nullptr)
        free(ctx_.input_attrs);
    if (ctx_.output_attrs != nullptr)
        free(ctx_.output_attrs);
    if (ctx_.rknn_ctx != 0)
        rknn_destroy(ctx_.rknn_ctx);
    ctx_.input_attrs = nullptr;
    ctx_.output_attrs = nullptr;
    ctx_.rknn_ctx = 0;
    loaded_ = false;
    labels_.clear();
    input_buf_.clear();
}

bool PpyoloeDetector::send_udp_json_(const std::vector<PredictResult> &detections) const { return true; }
std::string PpyoloeDetector::build_udp_json_(const std::vector<PredictResult> &detections) const { return ""; }

bool PpyoloeDetector::infer(const cv::Mat &bgr_image, std::vector<PredictResult> &detections)
{
    detections.clear();
    if (!loaded_ || bgr_image.empty())
        return false;

    static int prof_count = 0;
    static double prof_pre_ms = 0, prof_input_ms = 0, prof_npu_ms = 0, prof_output_ms = 0, prof_post_ms = 0;
    static auto prof_fps_t0 = std::chrono::steady_clock::now();
    auto t0 = std::chrono::steady_clock::now();

    const int num_elements = ctx_.model_width * ctx_.model_height * ctx_.model_channel;
    if (input_buf_.size() != static_cast<std::size_t>(num_elements))
        input_buf_.resize(static_cast<std::size_t>(num_elements));

    // 1. 预处理
    cv::Mat small_bgr;
    cv::resize(bgr_image, small_bgr, cv::Size(ctx_.model_width, ctx_.model_height), 0, 0, cv::INTER_LINEAR);
    resized_ = cv::Mat(ctx_.model_height, ctx_.model_width, CV_8UC3, input_buf_.data());
    cv::cvtColor(small_bgr, resized_, cv::COLOR_BGR2RGB);

    auto t1 = std::chrono::steady_clock::now();

    // 2. 安全的输入设定 (让驱动去处理 UINT8 -> INT8 的转换)
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8; // 极其重要：明确告诉底层传入的是 0~255 的图像
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].size = static_cast<uint32_t>(num_elements);
    inputs[0].buf = input_buf_.data();

    if (rknn_inputs_set(ctx_.rknn_ctx, 1, inputs) < 0)
        return false;

    auto t2 = std::chrono::steady_clock::now();

    // 3. 推理
    if (rknn_run(ctx_.rknn_ctx, nullptr) < 0)
        return false;

    auto t3 = std::chrono::steady_clock::now();

    // 4. 获取输出 (强制转换为浮点数以供后处理)
    rknn_output outputs[ctx_.io_num.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < ctx_.io_num.n_output; ++i)
    {
        outputs[i].index = i;
        outputs[i].want_float = 1;
    }

    if (rknn_outputs_get(ctx_.rknn_ctx, ctx_.io_num.n_output, outputs, NULL) < 0)
        return false;

    auto t4 = std::chrono::steady_clock::now();

    // 5. 后处理解码
    ppyoloe_post_process(&ctx_, outputs, bgr_image.cols, bgr_image.rows, config_.confidence_threshold, config_.nms_threshold, detections);
    rknn_outputs_release(ctx_.rknn_ctx, ctx_.io_num.n_output, outputs);

    auto t5 = std::chrono::steady_clock::now();

    using namespace std::chrono;
    prof_pre_ms += duration_cast<duration<double, std::milli>>(t1 - t0).count();
    prof_input_ms += duration_cast<duration<double, std::milli>>(t2 - t1).count();
    prof_npu_ms += duration_cast<duration<double, std::milli>>(t3 - t2).count();
    prof_output_ms += duration_cast<duration<double, std::milli>>(t4 - t3).count();
    prof_post_ms += duration_cast<duration<double, std::milli>>(t5 - t4).count();
    ++prof_count;

    if (prof_count >= 20)
    {
        auto prof_fps_t1 = std::chrono::steady_clock::now();
        double prof_elapsed = std::chrono::duration<double>(prof_fps_t1 - prof_fps_t0).count();
        double prof_fps = prof_elapsed > 0.0 ? prof_count / prof_elapsed : 0.0;
        if (config_.verbose)
            printf("[infer detail] fps=%.1f pre=%.2fms input=%.2fms npu=%.2fms output=%.2fms post=%.2fms | total=%.2fms\n",
                   prof_fps,
                   prof_pre_ms / prof_count, prof_input_ms / prof_count, prof_npu_ms / prof_count, prof_output_ms / prof_count, prof_post_ms / prof_count,
                   (prof_pre_ms + prof_input_ms + prof_npu_ms + prof_output_ms + prof_post_ms) / prof_count);
        prof_count = 0;
        prof_pre_ms = prof_input_ms = prof_npu_ms = prof_output_ms = prof_post_ms = 0.0;
        prof_fps_t0 = prof_fps_t1;
    }
    return true;
}

bool PpyoloeDetector::is_loaded() const { return loaded_; }