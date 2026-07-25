#include "ppyoloe_postprocess.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <utility>
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

    static const char *kLabels[11] = {"stop", "go", "guide_road", "guide_speed", "light_green", "light_red", "light_yellow", "zebraline", "gold", "human", "car"};

    static inline float clampf(float v, float lo, float hi)
    {
        return std::max(lo, std::min(v, hi));
    }

    static inline float box_iou(const Candidate &a, const Candidate &b)
    {
        const float xx1 = std::max(a.x1, b.x1);
        const float yy1 = std::max(a.y1, b.y1);
        const float xx2 = std::min(a.x2, b.x2);
        const float yy2 = std::min(a.y2, b.y2);
        const float w = std::max(0.0f, xx2 - xx1);
        const float h = std::max(0.0f, yy2 - yy1);
        const float inter = w * h;
        const float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
        const float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
        const float uni = area_a + area_b - inter;
        return (uni <= 0.0f) ? 0.0f : (inter / uni);
    }

    // PicoDet 专属的扁平化 DFL 极速解码// PicoDet 专属的扁平化 DFL 极速解码 (修复 NaN 崩溃版)
    static inline void dfl_decode_picodet(const float *pos_data, int bins, int idx, float out_dist[4])
    {
        const float *base = pos_data + idx * (4 * bins);
        for (int side = 0; side < 4; ++side)
        {
            const float *side_base = base + side * bins;
            float max_v = side_base[0];
            for (int k = 1; k < bins; ++k)
            {
                if (side_base[k] > max_v)
                    max_v = side_base[k];
            }

            float exp_sum = 0.0f;
            float acc = 0.0f;
            for (int k = 0; k < bins; ++k)
            {
                float val = side_base[k] - max_v;
                if (val < -9.0f)
                    continue; // 快速剪枝

                // =========================================================
                // 终极修复 2：彻底废除 Fast Exp 黑魔法，使用安全的 std::exp
                // 杜绝 ARM 编译器产生 NaN 导致全屏大框
                // =========================================================
                float e = std::exp(val);

                exp_sum += e;
                acc += e * static_cast<float>(k);
            }
            out_dist[side] = (exp_sum > 0.0f) ? (acc / exp_sum) : 0.0f;
        }
    }

    static void nms_per_class(const std::vector<Candidate> &cands, float nms_threshold, int img_w, int img_h, std::vector<PredictResult> &detections)
    {
        std::vector<Candidate> selected;
        for (int cid = 0; cid < 11; ++cid)
        {
            std::vector<int> idxs;
            for (int i = 0; i < static_cast<int>(cands.size()); ++i)
                if (cands[i].class_id == cid)
                    idxs.push_back(i);

            if (idxs.empty())
                continue;

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
            const Candidate &c = selected[i];
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
}

int ppyoloe_init_post_process() { return 0; }
void ppyoloe_deinit_post_process() {}
char *ppyoloe_cls_to_name(int cls_id) { return (cls_id < 0 || cls_id >= 11) ? const_cast<char *>("null") : const_cast<char *>(kLabels[cls_id]); }

int ppyoloe_post_process(ppyoloe_app_context_t *app_ctx, void *outputs, int img_width, int img_height, float conf_threshold, float nms_threshold, std::vector<PredictResult> &detections)
{
    if (app_ctx == nullptr || outputs == nullptr)
        return -1;
    detections.clear();

    if (app_ctx->io_num.n_output != 8)
    {
        std::fprintf(stderr, "[ERROR] PicoDet expects exactly 8 outputs, got %d\n", app_ctx->io_num.n_output);
        return -1;
    }

    rknn_output *out = static_cast<rknn_output *>(outputs);
    std::vector<Candidate> candidates;
    candidates.reserve(100);

    const int num_classes = 11;
    const int num_branches = 4;

    // =========================================================================
    // 动态匹配 8 个输出 Tensor（自适应 320/416/640 任意输入尺寸）
    // =========================================================================
    int cls_indices[4] = {-1, -1, -1, -1};
    int pos_indices[4] = {-1, -1, -1, -1};

    // 按 HW 值分组，同 HW 的为同一分支的 cls+pos 对
    std::vector<std::pair<int, int>> hw_pairs; // (hw, index)
    for (int i = 0; i < 8; ++i)
        hw_pairs.push_back({app_ctx->output_attrs[i].dims[1], i});

    // 按 HW 降序排列（大步长 = 小网格）
    std::sort(hw_pairs.begin(), hw_pairs.end(), [](const std::pair<int, int> &a, const std::pair<int, int> &b)
              { return a.first > b.first; });

    int branch = 0;
    for (int i = 0; i < 8; i += 2)
    {
        int idx_a = hw_pairs[i].second;
        int idx_b = hw_pairs[i + 1].second;
        int ca = app_ctx->output_attrs[idx_a].dims[2];
        int cb = app_ctx->output_attrs[idx_b].dims[2];

        if (ca == num_classes && cb == 32)
        {
            cls_indices[branch] = idx_a;
            pos_indices[branch] = idx_b;
        }
        else if (cb == num_classes && ca == 32)
        {
            cls_indices[branch] = idx_b;
            pos_indices[branch] = idx_a;
        }
        else
        {
            std::fprintf(stderr, "[ERROR] Unexpected output channels (%d, %d) at branch %d\n", ca, cb, branch);
            return -1;
        }
        ++branch;
    }

    // 检查是否所有分支都匹配成功
    for (int i = 0; i < 4; ++i)
    {
        if (cls_indices[i] == -1 || pos_indices[i] == -1)
        {
            std::fprintf(stderr, "[ERROR] Output tensor matching failed for branch %d\n", i);
            return -1;
        }
    }

    // 遍历 4 个尺度的分支
    for (int branch = 0; branch < num_branches; ++branch)
    {
        // 使用动态匹配到的正确索引
        int cls_idx = cls_indices[branch];
        int pos_idx = pos_indices[branch];

        const int hw = app_ctx->output_attrs[cls_idx].dims[1];
        const int grid_size = static_cast<int>(std::round(std::sqrt(hw)));

        const float stride_x = static_cast<float>(img_width) / static_cast<float>(grid_size);
        const float stride_y = static_cast<float>(img_height) / static_cast<float>(grid_size);

        const float *cls_data = static_cast<const float *>(out[cls_idx].buf);
        const float *pos_data = static_cast<const float *>(out[pos_idx].buf);

        for (int y = 0; y < grid_size; ++y)
        {
            for (int x = 0; x < grid_size; ++x)
            {
                const int idx = y * grid_size + x;

                float best_score = 0.0f;
                int best_cls = -1;
                for (int c = 0; c < num_classes; ++c)
                {
                    float score = cls_data[idx * num_classes + c];
                    if (score > best_score)
                    {
                        best_score = score;
                        best_cls = c;
                    }
                }

                if (best_cls < 0 || best_score < conf_threshold)
                    continue;

                float dist[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                dfl_decode_picodet(pos_data, 8, idx, dist);

                float cx = (static_cast<float>(x) + 0.5f) * stride_x;
                float cy = (static_cast<float>(y) + 0.5f) * stride_y;

                Candidate cand{};
                cand.x1 = cx - dist[0] * stride_x;
                cand.y1 = cy - dist[1] * stride_y;
                cand.x2 = cx + dist[2] * stride_x;
                cand.y2 = cy + dist[3] * stride_y;
                cand.class_id = best_cls;
                cand.score = best_score;

                candidates.push_back(cand);
            }
        }
    }

    if (!candidates.empty())
        nms_per_class(candidates, nms_threshold, img_width, img_height, detections);
    return 0;
}