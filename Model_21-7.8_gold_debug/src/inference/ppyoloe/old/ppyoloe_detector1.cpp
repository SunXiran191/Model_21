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
    // 读取整文件内容，用于加载标签文件。
    static std::string read_text_file(const std::string &path)
    {
        std::ifstream ifs(path.c_str());
        if (!ifs.is_open())
        {
            return std::string();
        }

        std::string text;
        std::string line;
        while (std::getline(ifs, line))
        {
            if (!line.empty() && line[line.size() - 1] == '\r')
            {
                line.erase(line.size() - 1);
            }
            if (!text.empty())
            {
                text.push_back('\n');
            }
            text += line;
        }
        return text;
    }

    static bool set_core_mask_if_needed(rknn_context ctx, int core_id, const char *tag)
    {
        if (core_id < 0)
        {
            return true;
        }

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
            std::cerr << "[" << tag << "] invalid npu_core=" << core_id << ", skip binding" << std::endl;
            return false;
        }

        const int ret = rknn_set_core_mask(ctx, mask);
        if (ret != RKNN_SUCC)
        {
            std::cerr << "[" << tag << "] rknn_set_core_mask failed, ret=" << ret << std::endl;
            return false;
        }
        return true;
    }

    // 打印 RKNN 张量信息，便于首次接入模型时确认输入输出布局。
    static void dump_tensor_attr(rknn_tensor_attr *attr)
    {
        printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%d, type=%d, qnt_type=%d, zp=%d, scale=%f\n",
               attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
               attr->n_elems, attr->size, attr->fmt, attr->type, attr->qnt_type, attr->zp, attr->scale);
    }

    // 将类别名中的引号和反斜杠转义，拼接 UDP JSON 时使用。
    static std::string json_escape(const std::string &in)
    {
        std::string out;
        out.reserve(in.size());
        for (char c : in)
        {
            if (c == '"' || c == '\\')
            {
                out.push_back('\\');
            }
            out.push_back(c);
        }
        return out;
    }
} // namespace

PpyoloeDetector::PpyoloeDetector()
    : loaded_(false), udp_socket_(-1), udp_ready_(false), input_mem_(nullptr), use_zero_copy_(false)
{
    ctx_ = ppyoloe_app_context_t{};
    memset(&udp_addr_, 0, sizeof(udp_addr_));
}

PpyoloeDetector::~PpyoloeDetector()
{
    unload();
}

// 读取标签文件；文件不存在或为空则加载失败。
bool PpyoloeDetector::load_labels_()
{
    labels_.clear();
    const std::string text = read_text_file(config_.labels_path);
    if (text.empty())
    {
        std::cerr << "[PPYOLOE] labels file not found or empty: " << config_.labels_path << std::endl;
        return false;
    }

    std::istringstream iss(text);
    std::string line;
    while (std::getline(iss, line))
    {
        if (!line.empty())
        {
            labels_.push_back(line);
        }
    }
    if (labels_.empty())
    {
        std::cerr << "[PPYOLOE] no valid labels in file" << std::endl;
        return false;
    }
    std::cout << "[PPYOLOE] loaded " << labels_.size() << " labels" << std::endl;
    return true;
}

// 加载 RKNN 模型并初始化推理上下文。
// 成功后可在循环里调用 infer()，失败时返回 false 并释放已申请资源。
bool PpyoloeDetector::load(const PpyoloeConfig &config)
{
    unload();
    config_ = config;

    if (!load_labels_())
    {
        return false;
    }

    std::ifstream ifs(config_.model_path.c_str(), std::ios::binary);
    if (!ifs.is_open())
    {
        std::cerr << "[PPYOLOE] cannot open model: " << config_.model_path << std::endl;
        return false;
    }
    std::vector<char> model_data((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    if (model_data.empty())
    {
        std::cerr << "[PPYOLOE] model file is empty: " << config_.model_path << std::endl;
        return false;
    }

    rknn_context ctx = 0;
    int ret = rknn_init(&ctx, model_data.data(), static_cast<uint32_t>(model_data.size()), 0, NULL);
    if (ret < 0)
    {
        std::cerr << "[PPYOLOE] rknn_init failed, ret=" << ret << std::endl;
        return false;
    }

    set_core_mask_if_needed(ctx, config_.npu_core, "PPYOLOE");

    rknn_input_output_num io_num{};
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC)
    {
        std::cerr << "[PPYOLOE] RKNN_QUERY_IN_OUT_NUM failed, ret=" << ret << std::endl;
        rknn_destroy(ctx);
        return false;
    }

    std::vector<rknn_tensor_attr> input_attrs(io_num.n_input);
    std::vector<rknn_tensor_attr> output_attrs(io_num.n_output);
    for (int i = 0; i < io_num.n_input; ++i)
    {
        memset(&input_attrs[i], 0, sizeof(rknn_tensor_attr));
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &input_attrs[i], sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            std::cerr << "[PPYOLOE] RKNN_QUERY_INPUT_ATTR failed, ret=" << ret << std::endl;
            rknn_destroy(ctx);
            return false;
        }
        dump_tensor_attr(&input_attrs[i]);
    }
    for (int i = 0; i < io_num.n_output; ++i)
    {
        memset(&output_attrs[i], 0, sizeof(rknn_tensor_attr));
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &output_attrs[i], sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            std::cerr << "[PPYOLOE] RKNN_QUERY_OUTPUT_ATTR failed, ret=" << ret << std::endl;
            rknn_destroy(ctx);
            return false;
        }
        dump_tensor_attr(&output_attrs[i]);
    }

    ctx_.rknn_ctx = ctx;
    ctx_.io_num = io_num;
    ctx_.input_attrs = static_cast<rknn_tensor_attr *>(malloc(io_num.n_input * sizeof(rknn_tensor_attr)));
    ctx_.output_attrs = static_cast<rknn_tensor_attr *>(malloc(io_num.n_output * sizeof(rknn_tensor_attr)));
    memcpy(ctx_.input_attrs, input_attrs.data(), io_num.n_input * sizeof(rknn_tensor_attr));
    memcpy(ctx_.output_attrs, output_attrs.data(), io_num.n_output * sizeof(rknn_tensor_attr));

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        ctx_.model_channel = input_attrs[0].dims[1];
        ctx_.model_height = input_attrs[0].dims[2];
        ctx_.model_width = input_attrs[0].dims[3];
    }
    else
    {
        ctx_.model_height = input_attrs[0].dims[1];
        ctx_.model_width = input_attrs[0].dims[2];
        ctx_.model_channel = input_attrs[0].dims[3];
    }

    // 保存模型实际的输入格式和类型，推理时匹配使用
    ctx_.input_fmt = input_attrs[0].fmt;
    ctx_.input_type = input_attrs[0].type;
    printf("[PPYOLOE] model input: %dx%dx%d fmt=%s type=%s\n",
           ctx_.model_width, ctx_.model_height, ctx_.model_channel,
           ctx_.input_fmt == RKNN_TENSOR_NCHW ? "NCHW" : "NHWC",
           ctx_.input_type == RKNN_TENSOR_UINT8 ? "UINT8" : (ctx_.input_type == RKNN_TENSOR_FLOAT32 ? "FP32" : "OTHER"));

    if (config_.enable_udp && config_.udp_port > 0)
    {
        udp_socket_ = socket(AF_INET, SOCK_DGRAM, 0);
        if (udp_socket_ < 0)
        {
            std::cerr << "[PPYOLOE] failed to create UDP socket" << std::endl;
            unload();
            return false;
        }
        udp_addr_.sin_family = AF_INET;
        udp_addr_.sin_port = htons(static_cast<uint16_t>(config_.udp_port));
        if (inet_pton(AF_INET, config_.udp_ip.c_str(), &udp_addr_.sin_addr) != 1)
        {
            std::cerr << "[PPYOLOE] invalid UDP target: " << config_.udp_ip << std::endl;
            unload();
            return false;
        }
        udp_ready_ = true;
    }

    // 创建 zero-copy non-cacheable 输入内存，消除 rknn_inputs_set 的 cache flush 开销
    {
        const uint32_t mem_size = ctx_.input_attrs[0].size_with_stride > 0
                                      ? ctx_.input_attrs[0].size_with_stride
                                      : ctx_.input_attrs[0].size;
        input_mem_ = rknn_create_mem2(ctx, static_cast<uint64_t>(mem_size), RKNN_FLAG_MEMORY_NON_CACHEABLE);
        if (input_mem_ != nullptr)
        {
            use_zero_copy_ = true;
            rknn_tensor_attr attr = ctx_.input_attrs[0];
            attr.pass_through = 0;  // 允许驱动做格式转换
            int ret = rknn_set_io_mem(ctx, input_mem_, &attr);
            if (ret != RKNN_SUCC)
            {
                std::cerr << "[PPYOLOE] rknn_set_io_mem failed, ret=" << ret << ", fallback to copy mode" << std::endl;
                rknn_destroy_mem(ctx, input_mem_);
                input_mem_ = nullptr;
                use_zero_copy_ = false;
            }
            else
            {
                printf("[PPYOLOE] zero-copy input enabled (non-cacheable)\n");
            }
        }
        if (!use_zero_copy_)
        {
            // Fallback: 预分配 UINT8 输入缓冲区
            const int num_elements = ctx_.model_width * ctx_.model_height * ctx_.model_channel;
            input_buf_.resize(static_cast<std::size_t>(num_elements));
        }
    }

    loaded_ = true;
    return true;
}

// 释放模型、标签、输出上下文和 UDP socket。
void PpyoloeDetector::unload()
{
    if (ctx_.input_attrs != nullptr)
    {
        free(ctx_.input_attrs);
        ctx_.input_attrs = nullptr;
    }
    if (ctx_.output_attrs != nullptr)
    {
        free(ctx_.output_attrs);
        ctx_.output_attrs = nullptr;
    }
    if (ctx_.rknn_ctx != 0)
    {
        rknn_destroy(ctx_.rknn_ctx);
        ctx_.rknn_ctx = 0;
    }
    if (udp_socket_ >= 0)
    {
        close(udp_socket_);
        udp_socket_ = -1;
    }
    udp_ready_ = false;
    loaded_ = false;
    labels_.clear();
    if (input_mem_ != nullptr)
    {
        rknn_destroy_mem(ctx_.rknn_ctx, input_mem_);
        input_mem_ = nullptr;
    }
    use_zero_copy_ = false;
    input_buf_.clear();
    rgb_image_.release();
    resized_.release();
}

// 按当前配置把检测结果发送到 UDP 目标。
// 如果 disable UDP，则视为成功返回，方便仅在本地消费结果。
bool PpyoloeDetector::send_udp_json_(const std::vector<PredictResult> &detections) const
{
    if (!udp_ready_ || udp_socket_ < 0)
    {
        return true;
    }

    const std::string payload = build_udp_json_(detections);
    const ssize_t sent = sendto(udp_socket_,
                                payload.data(),
                                payload.size(),
                                0,
                                reinterpret_cast<const sockaddr *>(&udp_addr_),
                                sizeof(udp_addr_));
    if (sent < 0)
    {
        std::cerr << "[PPYOLOE] sendto failed" << std::endl;
        return false;
    }
    return true;
}

// 组装与 icar 侧兼容的检测结果 JSON：count + objects[]。
std::string PpyoloeDetector::build_udp_json_(const std::vector<PredictResult> &detections) const
{
    std::ostringstream os;
    os.setf(std::ios::fixed);
    os << std::setprecision(3);
    os << "{\"count\":" << detections.size() << ",\"objects\":";
    os << '[';
    for (std::size_t i = 0; i < detections.size(); ++i)
    {
        if (i > 0)
        {
            os << ',';
        }
        const PredictResult &det = detections[i];
        const std::string class_name = (det.class_id >= 0 && det.class_id < static_cast<int>(labels_.size())) ? labels_[static_cast<std::size_t>(det.class_id)] : std::string("null");
        os << '{'
           << "\"class_id\":" << det.class_id << ','
           << "\"class_name\":\"" << json_escape(class_name) << "\","
           << "\"confidence\":" << det.score << ','
           << "\"x1\":" << det.x1 << ','
           << "\"y1\":" << det.y1 << ','
           << "\"x2\":" << det.x2 << ','
           << "\"y2\":" << det.y2
           << '}';
    }
    os << "]}";
    return os.str();
}

// 对单帧 BGR 图像执行完整推理流程：预处理、RKNN 推理、后处理、可选 UDP 发送。
bool PpyoloeDetector::infer(const cv::Mat &bgr_image, std::vector<PredictResult> &detections)
{
    detections.clear();
    if (!loaded_ || bgr_image.empty())
    {
        return false;
    }

    // ---- 精细化性能剖析 ----
    static int prof_count = 0;
    static double prof_pre_ms = 0, prof_input_ms = 0, prof_npu_ms = 0;
    static double prof_output_ms = 0, prof_post_ms = 0;
    auto t0 = std::chrono::steady_clock::now();

    const int num_elements = ctx_.model_width * ctx_.model_height * ctx_.model_channel;

    // ---- 预处理：先resize(小图BGR)再cvtColor(小图→RGB)，大幅减少像素操作 ----
    try
    {
        cv::Mat small_bgr;

        if (bgr_image.channels() == 3)
        {
            cv::resize(bgr_image, small_bgr,
                       cv::Size(ctx_.model_width, ctx_.model_height), 0, 0, cv::INTER_LINEAR);
        }
        else if (bgr_image.channels() == 1)
        {
            cv::Mat tmp;
            cv::resize(bgr_image, tmp,
                       cv::Size(ctx_.model_width, ctx_.model_height), 0, 0, cv::INTER_LINEAR);
            cv::cvtColor(tmp, small_bgr, cv::COLOR_GRAY2BGR);
        }
        else
        {
            std::cerr << "[PPYOLOE] unsupported input channels=" << bgr_image.channels() << std::endl;
            return false;
        }

        if (use_zero_copy_)
        {
            cv::Mat input_wrapper(ctx_.model_height, ctx_.model_width, CV_8UC3,
                                  input_mem_->virt_addr);
            cv::cvtColor(small_bgr, input_wrapper, cv::COLOR_BGR2RGB);
        }
        else
        {
            if (input_buf_.size() != static_cast<std::size_t>(num_elements))
                input_buf_.resize(static_cast<std::size_t>(num_elements));
            resized_ = cv::Mat(ctx_.model_height, ctx_.model_width, CV_8UC3, input_buf_.data());
            cv::cvtColor(small_bgr, resized_, cv::COLOR_BGR2RGB);
        }
    }
    catch (const cv::Exception &e)
    {
        std::cerr << "[PPYOLOE] preprocess failed: " << e.what() << std::endl;
        return false;
    }

    auto t1 = std::chrono::steady_clock::now();

    rknn_input inputs[ctx_.io_num.n_input];
    rknn_output outputs[ctx_.io_num.n_output];
    memset(inputs, 0, sizeof(inputs));
    memset(outputs, 0, sizeof(outputs));

    if (!use_zero_copy_)
    {
        // Fallback: 传统 rknn_inputs_set（有 cache flush 开销）
        inputs[0].index = 0;
        inputs[0].type = RKNN_TENSOR_UINT8;
        inputs[0].fmt = ctx_.input_fmt;
        inputs[0].size = static_cast<uint32_t>(num_elements);
        inputs[0].buf = input_buf_.data();

        if (rknn_inputs_set(ctx_.rknn_ctx, ctx_.io_num.n_input, inputs) < 0)
        {
            std::cerr << "[PPYOLOE] rknn_inputs_set failed" << std::endl;
            return false;
        }
    }
    // zero-copy 模式：已在 load() 中通过 rknn_set_io_mem 绑定，无需额外操作

    auto t2 = std::chrono::steady_clock::now();

    int ret = rknn_run(ctx_.rknn_ctx, nullptr);
    if (ret < 0)
    {
        std::cerr << "[PPYOLOE] rknn_run failed" << std::endl;
        return false;
    }

    auto t3 = std::chrono::steady_clock::now();

    for (int i = 0; i < ctx_.io_num.n_output; ++i)
    {
        outputs[i].index = i;
        outputs[i].want_float = 1;
    }

    ret = rknn_outputs_get(ctx_.rknn_ctx, ctx_.io_num.n_output, outputs, NULL);
    if (ret < 0)
    {
        std::cerr << "[PPYOLOE] rknn_outputs_get failed" << std::endl;
        return false;
    }

    auto t4 = std::chrono::steady_clock::now();

    ret = ppyoloe_post_process(&ctx_, outputs, bgr_image.cols, bgr_image.rows, config_.confidence_threshold, config_.nms_threshold, detections);
    rknn_outputs_release(ctx_.rknn_ctx, ctx_.io_num.n_output, outputs);

    auto t5 = std::chrono::steady_clock::now();

    if (ret != 0)
    {
        return false;
    }

    if (!send_udp_json_(detections))
    {
        return false;
    }

    // 每秒打印一次精细化剖析
    using namespace std::chrono;
    prof_pre_ms += duration_cast<duration<double, std::milli>>(t1 - t0).count();
    prof_input_ms += duration_cast<duration<double, std::milli>>(t2 - t1).count();
    prof_npu_ms += duration_cast<duration<double, std::milli>>(t3 - t2).count();
    prof_output_ms += duration_cast<duration<double, std::milli>>(t4 - t3).count();
    prof_post_ms += duration_cast<duration<double, std::milli>>(t5 - t4).count();
    ++prof_count;

    if (prof_count >= 20)  // 约每秒输出一次
    {
        printf("[infer detail] pre=%.2fms input=%.2fms npu=%.2fms output=%.2fms post=%.2fms | total=%.2fms\n",
               prof_pre_ms / prof_count,
               prof_input_ms / prof_count,
               prof_npu_ms / prof_count,
               prof_output_ms / prof_count,
               prof_post_ms / prof_count,
               (prof_pre_ms + prof_input_ms + prof_npu_ms + prof_output_ms + prof_post_ms) / prof_count);
        prof_count = 0;
        prof_pre_ms = prof_input_ms = prof_npu_ms = prof_output_ms = prof_post_ms = 0.0;
    }

    return true;
}

bool PpyoloeDetector::is_loaded() const
{
    return loaded_;
}