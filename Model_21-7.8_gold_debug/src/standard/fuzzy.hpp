#pragma once

/// @file fuzzy.hpp
/// @brief 行人速度模糊控制器
///
/// 输入变量（3个）：
///   1. control_state  — NONE(0)/GO(1)/SLOW(2)/STOP(3)/FAR(4)
///   2. offset_ratio   — |行人x - 中线x| / near_threshold,  [0, ~2+]
///   3. distance_ratio — 行人Y / IMAGE_H,  [0,1], 值越大越近
///
/// 输出变量（1个）：
///   speed_scale — 速度缩放因子, [0.0, 1.0]
///
/// 推理方法：Mamdani (MIN-AND, MAX-OR, COG 去模糊化)

class FuzzyHumanSpeedController
{
public:
    FuzzyHumanSpeedController();

    /// @brief 从 config 加载模糊输出中心值（启动时调用一次）
    void loadConfig(const struct Config &config);

    /// @brief 核心接口：控制状态 + 偏移比例 + 纵向距离
    float compute(int ctrl_state, float offset_ratio, float distance_ratio);

    /// @brief 简化接口（offset和distance用默认中性值）
    float compute(int ctrl_state);

private:
    // ================ 输入隶属度函数 ================

    /// ControlState 模糊化：相邻状态三角隶属度有重叠，保证平滑过渡
    float mu_ctrl_none(float s) const;
    float mu_ctrl_go(float s) const;
    float mu_ctrl_slow(float s) const;
    float mu_ctrl_stop(float s) const;
    float mu_ctrl_far(float s) const;

    /// offset_ratio 模糊化
    float mu_offset_near(float r) const;   // 近区 (0 ~ 1.05)
    float mu_offset_medium(float r) const; // 过渡 (0.45 ~ 2.10)
    float mu_offset_far(float r) const;    // 远区 (>2.10)

    /// distance_ratio 模糊化（纵向距离，越大越远）
    float mu_dist_far(float d) const;    // 远 (>0.20)
    float mu_dist_medium(float d) const; // 中 (0.04 ~ 0.25)
    float mu_dist_close(float d) const;  // 近 (0 ~ 0.09)

    // ================ 输出隶属度函数 ================

    float mu_scale_stop(float x) const; // 中心 0.00
    float mu_scale_slow(float x) const; // 中心 0.35
    float mu_scale_keep(float x) const; // 中心 0.65
    float mu_scale_full(float x) const; // 中心 1.00

    // ================ 去模糊化 ================

    /// 重心法 (Center of Gravity)
    /// @param strengths  各规则激活强度数组
    /// @param n_rules    规则数量
    float defuzzify_cog(const float *strengths, int n_rules) const;

    // 输出模糊集中心值（运行时从 config 加载，构造时使用默认值）
    float C_STOP = 0.00f;
    float C_SLOW = 0.35f;
    float C_KEEP = 0.85f;
    float C_FULL = 1.0f;

    // 规则数量
    static constexpr int NUM_RULES = 9;

    /// 规则结论对应的输出中心索引
    /// rules_out_[i] ∈ {0=STOP, 1=SLOW, 2=KEEP, 3=FULL}
    int rules_out_[NUM_RULES];
};
