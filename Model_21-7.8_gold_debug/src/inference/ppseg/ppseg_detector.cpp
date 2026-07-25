#include "ppseg_detector.hpp"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iterator>

namespace
{
    static void dump_tensor_attr(const char *tag, const rknn_tensor_attr &attr)
    {
        // std::fprintf(stderr,
        //              "[PPSEG] %s index=%d name=%s n_dims=%d dims=[%d,%d,%d,%d] n_elems=%d size=%d fmt=%d type=%d qnt_type=%d zp=%d scale=%f\n",
        //              tag,
        //              attr.index,
        //              attr.name,
        //              attr.n_dims,
        //              attr.dims[0],
        //              attr.dims[1],
        //              attr.dims[2],
        //              attr.dims[3],
        //              attr.n_elems,
        //              attr.size,
        //              attr.fmt,
        //              attr.type,
        //              attr.qnt_type,
        //              attr.zp,
        //              attr.scale);
    }

    static bool tensor_to_float_vector(const rknn_tensor_attr &attr, const rknn_output &output, std::vector<float> &values)
    {
        values.clear();

        const int element_count = static_cast<int>(attr.n_elems);
        if (element_count <= 0 || output.buf == nullptr)
        {
            return false;
        }

        values.resize(static_cast<std::size_t>(element_count));

        if (output.want_float)
        {
            const float *src = static_cast<const float *>(output.buf);
            std::memcpy(values.data(), src, static_cast<std::size_t>(element_count) * sizeof(float));
            return true;
        }

        if (attr.type == RKNN_TENSOR_FLOAT32)
        {
            const float *src = static_cast<const float *>(output.buf);
            std::memcpy(values.data(), src, static_cast<std::size_t>(element_count) * sizeof(float));
            return true;
        }

        if (attr.type == RKNN_TENSOR_INT8)
        {
            const int8_t *src = static_cast<const int8_t *>(output.buf);
            if (attr.qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC)
            {
                for (int i = 0; i < element_count; ++i)
                {
                    values[static_cast<std::size_t>(i)] = (static_cast<float>(src[i]) - static_cast<float>(attr.zp)) * attr.scale;
                }
                return true;
            }
            if (attr.qnt_type == RKNN_TENSOR_QNT_DFP)
            {
                const float scale = std::ldexp(1.0f, -static_cast<int>(attr.fl));
                for (int i = 0; i < element_count; ++i)
                {
                    values[static_cast<std::size_t>(i)] = static_cast<float>(src[i]) * scale;
                }
                return true;
            }
            for (int i = 0; i < element_count; ++i)
            {
                values[static_cast<std::size_t>(i)] = static_cast<float>(src[i]);
            }
            return true;
        }

        if (attr.type == RKNN_TENSOR_UINT8)
        {
            const uint8_t *src = static_cast<const uint8_t *>(output.buf);
            if (attr.qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC)
            {
                for (int i = 0; i < element_count; ++i)
                {
                    values[static_cast<std::size_t>(i)] = (static_cast<float>(src[i]) - static_cast<float>(attr.zp)) * attr.scale;
                }
                return true;
            }
            if (attr.qnt_type == RKNN_TENSOR_QNT_DFP)
            {
                const float scale = std::ldexp(1.0f, -static_cast<int>(attr.fl));
                for (int i = 0; i < element_count; ++i)
                {
                    values[static_cast<std::size_t>(i)] = static_cast<float>(src[i]) * scale;
                }
                return true;
            }
            for (int i = 0; i < element_count; ++i)
            {
                values[static_cast<std::size_t>(i)] = static_cast<float>(src[i]);
            }
            return true;
        }

        return false;
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
            std::fprintf(stderr, "[%s] invalid npu_core=%d, skip binding\n", tag, core_id);
            return false;
        }

        const int ret = rknn_set_core_mask(ctx, mask);
        if (ret != RKNN_SUCC)
        {
            std::fprintf(stderr, "[%s] rknn_set_core_mask failed, ret=%d\n", tag, ret);
            return false;
        }
        return true;
    }
} // namespace

PpsegDetector::PpsegDetector()
{
    memset(&ctx_, 0, sizeof(ctx_));
    ctx_.rknn_ctx = 0;
    ctx_.input_attrs = nullptr;
    ctx_.output_attrs = nullptr;
    loaded_ = false;
    input_size_ = 0;
}

PpsegDetector::~PpsegDetector()
{
    unload();
}

bool PpsegDetector::load(const std::string &model_path, int img_size, int npu_core)
{
    unload();
    model_path_ = model_path;

    std::ifstream ifs(model_path.c_str(), std::ios::binary);
    if (!ifs.is_open())
    {
        std::fprintf(stderr, "[PPSEG] cannot open model: %s\n", model_path.c_str());
        return false;
    }
    std::vector<char> model_data((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    if (model_data.empty())
    {
        std::fprintf(stderr, "[PPSEG] model file is empty: %s\n", model_path.c_str());
        return false;
    }

    rknn_context ctx = 0;
    int ret = rknn_init(&ctx, static_cast<void *>(model_data.data()), static_cast<uint32_t>(model_data.size()), 0, NULL);
    if (ret < 0)
    {
        std::fprintf(stderr, "[PPSEG] rknn_init failed, ret=%d\n", ret);
        return false;
    }

    set_core_mask_if_needed(ctx, npu_core, "PPSEG");

    rknn_input_output_num io_num{};
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC)
    {
        std::fprintf(stderr, "[PPSEG] RKNN_QUERY_IN_OUT_NUM failed, ret=%d\n", ret);
        rknn_destroy(ctx);
        return false;
    }

    ctx_.rknn_ctx = ctx;
    ctx_.io_num = io_num;
    ctx_.input_attrs = static_cast<rknn_tensor_attr *>(malloc(io_num.n_input * sizeof(rknn_tensor_attr)));
    ctx_.output_attrs = static_cast<rknn_tensor_attr *>(malloc(io_num.n_output * sizeof(rknn_tensor_attr)));

    std::vector<rknn_tensor_attr> in_attrs(io_num.n_input);
    std::vector<rknn_tensor_attr> out_attrs(io_num.n_output);
    for (int i = 0; i < io_num.n_input; ++i)
    {
        memset(&in_attrs[i], 0, sizeof(rknn_tensor_attr));
        in_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &in_attrs[i], sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            std::fprintf(stderr, "[PPSEG] RKNN_QUERY_INPUT_ATTR failed, ret=%d\n", ret);
            rknn_destroy(ctx);
            return false;
        }
        dump_tensor_attr("input", in_attrs[i]);
    }

    for (int i = 0; i < io_num.n_output; ++i)
    {
        memset(&out_attrs[i], 0, sizeof(rknn_tensor_attr));
        out_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &out_attrs[i], sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            std::fprintf(stderr, "[PPSEG] RKNN_QUERY_OUTPUT_ATTR failed, ret=%d\n", ret);
            rknn_destroy(ctx);
            return false;
        }
        dump_tensor_attr("output", out_attrs[i]);
    }

    memcpy(ctx_.input_attrs, in_attrs.data(), io_num.n_input * sizeof(rknn_tensor_attr));
    memcpy(ctx_.output_attrs, out_attrs.data(), io_num.n_output * sizeof(rknn_tensor_attr));

    // 假设模型的输入为第一个输入
    rknn_tensor_attr &ia = ctx_.input_attrs[0];
    if (ia.fmt == RKNN_TENSOR_NCHW)
    {
        ctx_.model_channel = ia.dims[1];
        ctx_.model_height = ia.dims[2];
        ctx_.model_width = ia.dims[3];
    }
    else
    {
        ctx_.model_height = ia.dims[1];
        ctx_.model_width = ia.dims[2];
        ctx_.model_channel = ia.dims[3];
    }

    input_size_ = ctx_.model_width * ctx_.model_height * ctx_.model_channel;

    loaded_ = true;
    return true;
}

void PpsegDetector::unload()
{
    if (ctx_.output_attrs != nullptr)
    {
        free(ctx_.output_attrs);
        ctx_.output_attrs = nullptr;
    }
    if (ctx_.input_attrs != nullptr)
    {
        free(ctx_.input_attrs);
        ctx_.input_attrs = nullptr;
    }
    if (ctx_.rknn_ctx != 0)
    {
        rknn_destroy(ctx_.rknn_ctx);
        ctx_.rknn_ctx = 0;
    }
    loaded_ = false;
}

bool PpsegDetector::is_loaded() const
{
    return loaded_;
}

int PpsegDetector::model_width() const { return ctx_.model_width; }
int PpsegDetector::model_height() const { return ctx_.model_height; }
int PpsegDetector::model_channel() const { return ctx_.model_channel; }

// decode_output_to_mask：给定输出张量属性和浮点缓冲区（取第一个 batch），
// 解码并生成 CV_8UC1 的 mask。
// 兼容 Python 版 ppseg.py 的多种输出形状：2D / 3D / 4D。
bool PpsegDetector::decode_output_to_mask(const rknn_tensor_attr &attr, const float *buf, cv::Mat &mask)
{
    if (attr.n_dims <= 0)
    {
        return false;
    }

    // Python 版里先 squeeze，再按 ndim 处理。
    // 这里尽量复刻同样的逻辑。
    if (attr.n_dims == 2)
    {
        const int h = attr.dims[0];
        const int w = attr.dims[1];
        if (h <= 0 || w <= 0)
        {
            return false;
        }

        mask.create(h, w, CV_8UC1);

        float minv = buf[0];
        float maxv = buf[0];
        const int hw = h * w;
        for (int i = 0; i < hw; ++i)
        {
            const float v = buf[i];
            if (v < minv)
            {
                minv = v;
            }
            if (v > maxv)
            {
                maxv = v;
            }
        }

        const float threshold = (minv >= 0.0f && maxv <= 1.0f) ? 0.5f : 0.0f;
        for (int y = 0; y < h; ++y)
        {
            uint8_t *row = mask.ptr<uint8_t>(y);
            for (int x = 0; x < w; ++x)
            {
                row[x] = (buf[y * w + x] > threshold) ? 255u : 0u;
            }
        }
        return true;
    }

    if (attr.n_dims == 3)
    {
        // 兼容三种常见情况：
        // 1) [2, H, W] 或 [C, H, W]：通道优先
        // 2) [H, W, 2] 或 [H, W, C]：通道在最后
        // 3) [1, H, W] / [H, W, 1]：单通道
        const int d0 = attr.dims[0];
        const int d1 = attr.dims[1];
        const int d2 = attr.dims[2];

        if (d0 == 1)
        {
            const int h = d1;
            const int w = d2;
            if (h <= 0 || w <= 0)
            {
                return false;
            }
            mask.create(h, w, CV_8UC1);
            float minv = buf[0];
            float maxv = buf[0];
            const int hw = h * w;
            for (int i = 0; i < hw; ++i)
            {
                const float v = buf[i];
                if (v < minv)
                {
                    minv = v;
                }
                if (v > maxv)
                {
                    maxv = v;
                }
            }
            const float threshold = (minv >= 0.0f && maxv <= 1.0f) ? 0.5f : 0.0f;
            for (int y = 0; y < h; ++y)
            {
                uint8_t *row = mask.ptr<uint8_t>(y);
                for (int x = 0; x < w; ++x)
                {
                    row[x] = (buf[y * w + x] > threshold) ? 255u : 0u;
                }
            }
            return true;
        }

        if (d2 == 1)
        {
            const int h = d0;
            const int w = d1;
            if (h <= 0 || w <= 0)
            {
                return false;
            }
            mask.create(h, w, CV_8UC1);
            float minv = buf[0];
            float maxv = buf[0];
            const int hw = h * w;
            for (int i = 0; i < hw; ++i)
            {
                const float v = buf[i];
                if (v < minv)
                {
                    minv = v;
                }
                if (v > maxv)
                {
                    maxv = v;
                }
            }
            const float threshold = (minv >= 0.0f && maxv <= 1.0f) ? 0.5f : 0.0f;
            for (int y = 0; y < h; ++y)
            {
                uint8_t *row = mask.ptr<uint8_t>(y);
                for (int x = 0; x < w; ++x)
                {
                    row[x] = (buf[y * w + x] > threshold) ? 255u : 0u;
                }
            }
            return true;
        }

        if (d0 == 2)
        {
            const int c = d0;
            const int h = d1;
            const int w = d2;
            mask.create(h, w, CV_8UC1);
            const int hw = h * w;
            for (int y = 0; y < h; ++y)
            {
                uint8_t *row = mask.ptr<uint8_t>(y);
                for (int x = 0; x < w; ++x)
                {
                    int best_c = 0;
                    float best_v = buf[0 * hw + y * w + x];
                    for (int ch = 1; ch < c; ++ch)
                    {
                        const float v = buf[ch * hw + y * w + x];
                        if (v > best_v)
                        {
                            best_v = v;
                            best_c = ch;
                        }
                    }
                    row[x] = (best_c == 1) ? 255u : 0u;
                }
            }
            return true;
        }

        if (d2 == 2)
        {
            const int h = d0;
            const int w = d1;
            const int c = d2;
            mask.create(h, w, CV_8UC1);
            for (int y = 0; y < h; ++y)
            {
                uint8_t *row = mask.ptr<uint8_t>(y);
                for (int x = 0; x < w; ++x)
                {
                    int best_c = 0;
                    float best_v = buf[(y * w + x) * c + 0];
                    for (int ch = 1; ch < c; ++ch)
                    {
                        const float v = buf[(y * w + x) * c + ch];
                        if (v > best_v)
                        {
                            best_v = v;
                            best_c = ch;
                        }
                    }
                    row[x] = (best_c == 1) ? 255u : 0u;
                }
            }
            return true;
        }

        return false;
    }

    // 4 维及以上：按原来的 4D 逻辑处理，只取前 4 维。
    int n = attr.dims[0];
    int c = attr.dims[1];
    int h = attr.dims[2];
    int w = attr.dims[3];
    if (attr.fmt != RKNN_TENSOR_NCHW)
    {
        n = attr.dims[0];
        h = attr.dims[1];
        w = attr.dims[2];
        c = attr.dims[3];
    }

    if (n <= 0 || h <= 0 || w <= 0 || c <= 0)
    {
        return false;
    }

    mask.create(h, w, CV_8UC1);

    if (c == 1)
    {
        // 单通道概率/logit：按阈值二值化为前景/背景
        float minv = buf[0];
        float maxv = buf[0];
        const int hw = h * w;
        for (int i = 0; i < hw; ++i)
        {
            float v = buf[i];
            if (v < minv)
                minv = v;
            if (v > maxv)
                maxv = v;
        }
        float threshold = 0.0f;
        if (minv >= 0.0f && maxv <= 1.0f)
            threshold = 0.5f;

        for (int y = 0; y < h; ++y)
        {
            uint8_t *row = mask.ptr<uint8_t>(y);
            for (int x = 0; x < w; ++x)
            {
                float v = buf[y * w + x];
                row[x] = (v > threshold) ? 255u : 0u;
            }
        }
        return true;
    }

    // 若通道数 > 1，则可能为 NCHW（通道优先）或 NHWC（通道在最后）。
    // NCHW 时，缓冲区按通道块（每通道 h*w）连续存放；NHWC 时，按像素依次存放各通道值。
    // 下面根据 attr.fmt 分别处理两种布局。

    if (attr.fmt == RKNN_TENSOR_NCHW)
    {
        const int hw = h * w;
        for (int y = 0; y < h; ++y)
        {
            uint8_t *row = mask.ptr<uint8_t>(y);
            for (int x = 0; x < w; ++x)
            {
                int best_c = 0;
                float best_v = buf[0 * hw + y * w + x];
                for (int ch = 1; ch < c; ++ch)
                {
                    float v = buf[ch * hw + y * w + x];
                    if (v > best_v)
                    {
                        best_v = v;
                        best_c = ch;
                    }
                }
                // 若为二分类（c==2），输出 0/255 表示类别
                if (c == 2)
                {
                    row[x] = (best_c == 1) ? 255u : 0u;
                }
                else
                {
                    // 多类分割：存储类别 id（范围 0..255）
                    int cid = best_c;
                    if (cid < 0)
                        cid = 0;
                    if (cid > 255)
                        cid = 255;
                    row[x] = static_cast<uint8_t>(cid);
                }
            }
        }
        return true;
    }

    // NHWC 情况：缓冲区按行主序，每个像素包含 c 个通道值
    if (attr.fmt != RKNN_TENSOR_NCHW)
    {
        for (int y = 0; y < h; ++y)
        {
            uint8_t *row = mask.ptr<uint8_t>(y);
            for (int x = 0; x < w; ++x)
            {
                int best_c = 0;
                float best_v = buf[(y * w + x) * c + 0];
                for (int ch = 1; ch < c; ++ch)
                {
                    float v = buf[(y * w + x) * c + ch];
                    if (v > best_v)
                    {
                        best_v = v;
                        best_c = ch;
                    }
                }
                if (c == 2)
                {
                    row[x] = (best_c == 1) ? 255u : 0u;
                }
                else
                {
                    int cid = best_c;
                    if (cid < 0)
                        cid = 0;
                    if (cid > 255)
                        cid = 255;
                    row[x] = static_cast<uint8_t>(cid);
                }
            }
        }
        return true;
    }

    return false;
}

bool PpsegDetector::infer(const cv::Mat &bgr_image, cv::Mat &out_mask, bool resize_to_input)
{
    if (!loaded_ || bgr_image.empty())
    {
        std::fprintf(stderr, "[PPSEG] infer skipped: loaded=%d empty=%d\n", loaded_ ? 1 : 0, bgr_image.empty() ? 1 : 0);
        return false;
    }

    // 准备输入缓冲区
    cv::Mat rgb;
    cv::cvtColor(bgr_image, rgb, cv::COLOR_BGR2RGB);
    cv::Mat resized(ctx_.model_height, ctx_.model_width, CV_8UC3);
    cv::resize(rgb, resized, resized.size(), 0, 0, cv::INTER_LINEAR);

    std::vector<unsigned char> input_buf(static_cast<size_t>(input_size_));
    std::memcpy(input_buf.data(), resized.data, static_cast<size_t>(input_size_));

    rknn_input inputs[ctx_.io_num.n_input];
    rknn_output outputs[ctx_.io_num.n_output];
    memset(inputs, 0, sizeof(inputs));
    memset(outputs, 0, sizeof(outputs));

    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].size = static_cast<uint32_t>(input_buf.size());
    inputs[0].buf = input_buf.data();

    int ret = rknn_inputs_set(ctx_.rknn_ctx, ctx_.io_num.n_input, inputs);
    if (ret < 0)
    {
        std::fprintf(stderr, "[PPSEG] rknn_inputs_set failed, ret=%d\n", ret);
        return false;
    }

    const std::chrono::steady_clock::time_point t_run_start = std::chrono::steady_clock::now();
    ret = rknn_run(ctx_.rknn_ctx, nullptr);
    const std::chrono::steady_clock::time_point t_run_end = std::chrono::steady_clock::now();
    if (ret < 0)
    {
        std::fprintf(stderr, "[PPSEG] rknn_run failed, ret=%d\n", ret);
        return false;
    }

    for (int i = 0; i < ctx_.io_num.n_output; ++i)
    {
        outputs[i].index = i;
        outputs[i].want_float = 1;
    }

    ret = rknn_outputs_get(ctx_.rknn_ctx, ctx_.io_num.n_output, outputs, NULL);
    const std::chrono::steady_clock::time_point t_get_end = std::chrono::steady_clock::now();
    if (ret < 0)
    {
        std::fprintf(stderr, "[PPSEG] rknn_outputs_get failed, ret=%d\n", ret);
        return false;
    }

    const double run_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t_run_end - t_run_start).count();
    const double get_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t_get_end - t_run_end).count();
    const double total_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t_get_end - t_run_start).count();
    static int s_frame_id = 0;
    ++s_frame_id;
    if ((s_frame_id % 30) == 0 || run_ms > 100.0 || get_ms > 100.0)
    {
        std::fprintf(stderr, "[PPSEG] timing: run=%.2f ms get=%.2f ms total=%.2f ms\n",
                     run_ms, get_ms, total_ms);
    }

    if (ctx_.io_num.n_output >= 1)
    {
        std::fprintf(stderr,
                     "[PPSEG] output[0] size=%u want_float=%d type=%d qnt_type=%d n_dims=%d\n",
                     outputs[0].size,
                     outputs[0].want_float,
                     ctx_.output_attrs[0].type,
                     ctx_.output_attrs[0].qnt_type,
                     ctx_.output_attrs[0].n_dims);
    }

    // 分割模型通常只有一个输出张量，这里默认取第 0 个输出
    bool ok = false;
    if (ctx_.io_num.n_output >= 1)
    {
        std::vector<float> float_output;
        if (!tensor_to_float_vector(ctx_.output_attrs[0], outputs[0], float_output))
        {
            std::fprintf(stderr, "[PPSEG] tensor_to_float_vector failed: type=%d qnt_type=%d size=%u n_elems=%d\n",
                         ctx_.output_attrs[0].type,
                         ctx_.output_attrs[0].qnt_type,
                         outputs[0].size,
                         ctx_.output_attrs[0].n_elems);
        }
        else
        {
            ok = decode_output_to_mask(ctx_.output_attrs[0], float_output.data(), out_mask);
        }
    }

    rknn_outputs_release(ctx_.rknn_ctx, ctx_.io_num.n_output, outputs);

    if (!ok)
    {
        std::fprintf(stderr, "[PPSEG] decode_output_to_mask failed\n");
        return false;
    }

    if (resize_to_input && (out_mask.cols != bgr_image.cols || out_mask.rows != bgr_image.rows))
    {
        cv::Mat scaled;
        cv::resize(out_mask, scaled, cv::Size(bgr_image.cols, bgr_image.rows), 0, 0, cv::INTER_NEAREST);
        out_mask = scaled;
    }

    return true;
}
