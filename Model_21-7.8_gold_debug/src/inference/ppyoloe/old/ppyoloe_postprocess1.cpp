#include "ppyoloe_postprocess.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace
{
    struct Candidate
    {
        float x1;
        float y1;
        float x2;
        float y2;
        float score;
        int class_id;
    };

    // ���֧�ֵ������������������������ڲ���������
    static const int kMaxClasses = 128;

    static inline float clampf(float v, float lo, float hi)
    {
        return std::max(lo, std::min(v, hi));
    }

    static inline float sigmoidf(float x)
    {
        // 快速稳定 sigmoid：使用与 DFL 相同的 exp 近似技巧
        if (x >= 0.0f)
        {
            if (x > 9.0f) return 1.0f;
            union { uint32_t i; float f; } v;
            v.i = static_cast<uint32_t>((1 << 23) * (1.4426950409f * (-x) + 126.93490512f));
            return 1.0f / (1.0f + v.f);
        }
        if (x < -9.0f) return 0.0f;
        union { uint32_t i; float f; } v;
        v.i = static_cast<uint32_t>((1 << 23) * (1.4426950409f * x + 126.93490512f));
        const float ex = v.f;
        return ex / (1.0f + ex);
    }

    static inline float box_iou(const Candidate &a, const Candidate &b)
    {
        const float xx1 = std::max(a.x1, b.x1);
        const float yy1 = std::max(a.y1, b.y1);
        const float xx2 = std::min(a.x2, b.x2);
        const float yy2 = std::min(a.y2, b.y2);
        const float w = std::max(0.0f, xx2 - xx1 + 1e-5f);
        const float h = std::max(0.0f, yy2 - yy1 + 1e-5f);
        const float inter = w * h;
        const float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
        const float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
        const float uni = area_a + area_b - inter;
        if (uni <= 0.0f)
            return 0.0f;
        return inter / uni;
    }

    // �������Ż��桿DFL ������� �� NCHW / channels?first ����
    // position ����: [C, H, W]���� position[channel * HW + y*W + x]
    static void dfl_decode_nchw(const float *position, int channels, int h, int w, int idx, float out_dist[4])
    {
        const int bins = channels / 4;
        const int hw = h * w;
        for (int side = 0; side < 4; ++side)
        {
            const float *base = position + side * bins * hw + idx;
            float max_v = base[0];
            for (int k = 1; k < bins; ++k)
            {
                const float v = base[k * hw];
                if (v > max_v)
                    max_v = v;
            }

            float exp_sum = 0.0f;
            float acc = 0.0f;
            for (int k = 0; k < bins; ++k)
            {
                float val = base[k * hw] - max_v;

                if (val < -9.0f)
                    continue;

                union
                {
                    uint32_t i;
                    float f;
                } v;
                v.i = static_cast<uint32_t>((1 << 23) * (1.4426950409f * val + 126.93490512f));
                const float e = v.f;

                exp_sum += e;
                acc += e * static_cast<float>(k);
            }
            out_dist[side] = (exp_sum > 0.0f) ? (acc / exp_sum) : 0.0f;
        }
    }

    // DFL ������� �� channels?last / չƽ���֣�PicoDet RKNN ���ã�
    // position ����: [HW, C]���� position[spatial * C + channel]
    static void dfl_decode_chlast(const float *position, int channels, int hw, int idx, float out_dist[4])
    {
        const int bins = channels / 4;
        for (int side = 0; side < 4; ++side)
        {
            const int ch_offset = side * bins;
            float max_v = position[idx * channels + ch_offset];
            for (int k = 1; k < bins; ++k)
            {
                const float v = position[idx * channels + ch_offset + k];
                if (v > max_v)
                    max_v = v;
            }

            float exp_sum = 0.0f;
            float acc = 0.0f;
            for (int k = 0; k < bins; ++k)
            {
                float val = position[idx * channels + ch_offset + k] - max_v;

                if (val < -9.0f)
                    continue;

                union
                {
                    uint32_t i;
                    float f;
                } v;
                v.i = static_cast<uint32_t>((1 << 23) * (1.4426950409f * val + 126.93490512f));
                const float e = v.f;

                exp_sum += e;
                acc += e * static_cast<float>(k);
            }
            out_dist[side] = (exp_sum > 0.0f) ? (acc / exp_sum) : 0.0f;
        }
    }

    static void nms_per_class(const std::vector<Candidate> &cands,
                              float nms_threshold,
                              int img_w,
                              int img_h,
                              int num_classes,
                              std::vector<PredictResult> &detections)
    {
        std::vector<Candidate> selected;
        for (int cid = 0; cid < num_classes; ++cid)
        {
            std::vector<int> idxs;
            for (int i = 0; i < static_cast<int>(cands.size()); ++i)
            {
                if (cands[i].class_id == cid)
                    idxs.push_back(i);
            }

            std::sort(idxs.begin(), idxs.end(), [&](int a, int b)
                      { return cands[a].score > cands[b].score; });

            std::vector<bool> removed(idxs.size(), false);
            for (std::size_t i = 0; i < idxs.size(); ++i)
            {
                if (removed[i])
                    continue;

                const Candidate &best = cands[idxs[i]];
                selected.push_back(best);

                for (std::size_t j = i + 1; j < idxs.size(); ++j)
                {
                    if (removed[j])
                        continue;
                    if (box_iou(best, cands[idxs[j]]) > nms_threshold)
                        removed[j] = true;
                }
            }
        }

        const int limit = std::min(static_cast<int>(selected.size()), 128);
        detections.reserve(limit);
        for (int i = 0; i < limit; ++i)
        {
            const Candidate &c = selected[static_cast<std::size_t>(i)];
            PredictResult result{};
            result.x1 = static_cast<int>(clampf(c.x1, 0.0f, static_cast<float>(img_w - 1)));
            result.y1 = static_cast<int>(clampf(c.y1, 0.0f, static_cast<float>(img_h - 1)));
            result.x2 = static_cast<int>(clampf(c.x2, 0.0f, static_cast<float>(img_w - 1)));
            result.y2 = static_cast<int>(clampf(c.y2, 0.0f, static_cast<float>(img_h - 1)));
            result.score = c.score;
            result.class_id = c.class_id;
            detections.push_back(result);
        }
    }
} // namespace

int ppyoloe_init_post_process() { return 0; }
void ppyoloe_deinit_post_process() {}

char *ppyoloe_cls_to_name(int cls_id)
{
    // ���������ַ�����ʾ�������ǩ�����ⲿ labels.txt �ṩ
    // �ú����������ڵ������������������������������Ӳ�����ǩ
    static char buf[32];
    if (cls_id < 0)
    {
        return const_cast<char *>("null");
    }
    std::snprintf(buf, sizeof(buf), "class_%d", cls_id);
    return buf;
}

int ppyoloe_post_process(ppyoloe_app_context_t *app_ctx,
                         void *outputs,
                         int img_width,
                         int img_height,
                         float conf_threshold,
                         float nms_threshold,
                         std::vector<PredictResult> &detections)
{
    if (app_ctx == nullptr || outputs == nullptr)
        return -1;
    detections.clear();

    const int n_out = app_ctx->io_num.n_output;
    rknn_output *out = static_cast<rknn_output *>(outputs);
    std::vector<Candidate> candidates;
    int num_classes = 0;

    // 使用缓存标志，首次推理自动检测是否需要 sigmoid
    bool need_sigmoid = app_ctx->need_sigmoid;

    // ===== 辅助：提取 tensor 的通道数和 HW =====
    auto get_tensor_chw = [&](int idx, int &ch, int &hw, int &h, int &w, bool &flat3d)
    {
        const rknn_tensor_attr &attr = app_ctx->output_attrs[idx];
        flat3d = (attr.n_dims == 3);
        if (flat3d)
        {
            hw = attr.dims[1];
            ch = attr.dims[2];
            const int side = static_cast<int>(std::sqrt(static_cast<float>(hw)));
            h = side;
            w = side;
        }
        else if (attr.fmt == RKNN_TENSOR_NCHW)
        {
            ch = attr.dims[1];
            h = attr.dims[2];
            w = attr.dims[3];
            hw = h * w;
        }
        else
        {
            h = attr.dims[1];
            w = attr.dims[2];
            ch = attr.dims[3];
            hw = h * w;
        }
    };

    // ===== 首次推理自动检测 sigmoid，后续复用缓存 =====
    auto detect_sigmoid = [&](int cls_idx, int cls_c)
    {
        if (app_ctx->sigmoid_detected || cls_c <= 0)
            return;
        const float *p = static_cast<const float *>(out[cls_idx].buf);
        for (int i = 0; i < std::min(cls_c * 4, 40); ++i)
        {
            if (p[i] > 1.5f || p[i] < -1.0f)
            {
                app_ctx->need_sigmoid = true;
                need_sigmoid = true;
                break;
            }
        }
    };

    // ===== 处理单个检测分支 =====
    auto process_branch = [&](int pos_idx, int cls_idx)
    {
        int pos_c, pos_hw, pos_h, pos_w, cls_c, cls_hw, cls_h, cls_w;
        bool flat3d, cls_flat;
        get_tensor_chw(pos_idx, pos_c, pos_hw, pos_h, pos_w, flat3d);
        get_tensor_chw(cls_idx, cls_c, cls_hw, cls_h, cls_w, cls_flat);

        if (cls_c > num_classes)
            num_classes = cls_c;

        detect_sigmoid(cls_idx, cls_c);

        const float *pos = static_cast<const float *>(out[pos_idx].buf);
        const float *cls = static_cast<const float *>(out[cls_idx].buf);

        // 预计算缩放因子（每个分支只需算一次）
        const float sx = static_cast<float>(img_width) / static_cast<float>(pos_w);
        const float sy = static_cast<float>(img_height) / static_cast<float>(pos_h);

        for (int y = 0; y < pos_h; ++y)
        {
            for (int x = 0; x < pos_w; ++x)
            {
                const int sp = y * pos_w + x;
                float dist[4] = {0, 0, 0, 0};

                if (flat3d)
                    dfl_decode_chlast(pos, pos_c, pos_hw, sp, dist);
                else
                    dfl_decode_nchw(pos, pos_c, pos_h, pos_w, sp, dist);

                float best = -std::numeric_limits<float>::infinity();
                int best_c = -1;
                if (cls_flat)
                {
                    const float *row = cls + sp * cls_c;
                    for (int c = 0; c < cls_c; ++c)
                    {
                        float v = row[c];
                        if (need_sigmoid)
                            v = sigmoidf(v);
                        if (v > best)
                        {
                            best = v;
                            best_c = c;
                        }
                    }
                }
                else
                {
                    for (int c = 0; c < cls_c; ++c)
                    {
                        float v = cls[c * pos_hw + sp];
                        if (need_sigmoid)
                            v = sigmoidf(v);
                        if (v > best)
                        {
                            best = v;
                            best_c = c;
                        }
                    }
                }

                if (best_c >= 0 && best >= conf_threshold)
                {
                    Candidate cand{};
                    cand.x1 = (static_cast<float>(x) + 0.5f - dist[0]) * sx;
                    cand.y1 = (static_cast<float>(y) + 0.5f - dist[1]) * sy;
                    cand.x2 = (static_cast<float>(x) + 0.5f + dist[2]) * sx;
                    cand.y2 = (static_cast<float>(y) + 0.5f + dist[3]) * sy;
                    cand.score = best;
                    cand.class_id = best_c;
                    candidates.push_back(cand);
                }
            }
        }
    };

    // ===== PicoDet��8 ������������� [cls��4, bbox��4]���� HW ��� =====
    if (n_out == 8)
    {
        struct TInfo
        {
            int idx, ch, hw;
        };
        TInfo t[8];
        for (int i = 0; i < 8; ++i)
        {
            int ch, hw_val, h, w;
            bool f;
            get_tensor_chw(i, ch, hw_val, h, w, f);
            t[i] = {i, ch, hw_val};
        }
        // ��ԣ�cls (ch%4!=0) �� bbox (ch%4==0, ��ͬ hw)
        for (int b = 0; b < 4; ++b)
        {
            int cls_i = -1, pos_i = -1, thw = -1;
            for (int i = 0; i < 8; ++i)
            {
                if (t[i].idx < 0)
                    continue;
                if (t[i].ch % 4 != 0)
                {
                    cls_i = t[i].idx;
                    thw = t[i].hw;
                    t[i].idx = -1;
                    break;
                }
            }
            for (int i = 0; i < 8; ++i)
            {
                if (t[i].idx < 0)
                    continue;
                if (t[i].ch % 4 == 0 && t[i].hw == thw)
                {
                    pos_i = t[i].idx;
                    t[i].idx = -1;
                    break;
                }
            }
            if (cls_i >= 0 && pos_i >= 0)
                process_branch(pos_i, cls_i);
        }
    }
    // ===== PP-YOLOE��9 ��������� [bbox, cls, obj] �� 3 =====
    else if (n_out == 9)
    {
        for (int b = 0; b < 3; ++b)
            process_branch(b * 3, b * 3 + 1); // pos=idx0, cls=idx1
    }
    // ===== PP-YOLOE ���壺6 ��������� [bbox, cls] �� 3 =====
    else if (n_out == 6)
    {
        for (int b = 0; b < 3; ++b)
            process_branch(b * 2, b * 2 + 1);
    }
    else
    {
        std::fprintf(stderr, "Unexpected n_output: %d\n", n_out);
        return -1;
    }

    // 首次推理后标记 sigmoid 检测完成，后续帧直接复用
    app_ctx->sigmoid_detected = true;

    if (!candidates.empty())
    {
        if (num_classes <= 0)
            num_classes = kMaxClasses;
        nms_per_class(candidates, nms_threshold, img_width, img_height, num_classes, detections);
    }
    return 0;
}