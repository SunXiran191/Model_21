#include "fuzzy.hpp"

#include <algorithm>
#include <cmath>

#include "../../res/configs/param.hpp"

// ============================================================================
// 工具函数
// ============================================================================

static float tri_mf(float x, float a, float b, float c)
{
    if (x == b)
        return 1.0f;
    if (x <= a || x >= c)
        return 0.0f;
    if (x < b)
        return (x - a) / (b - a);
    return (c - x) / (c - b);
}

// 梯形隶属度（左肩/右肩用）：x ≤ a → 0, a < x < b → 斜坡, b ≤ x ≤ c → 1, c < x < d → 斜坡, x ≥ d → 0
static float trap_mf(float x, float a, float b, float c, float d)
{
    if (x <= a || x >= d)
        return 0.0f;
    if (x >= b && x <= c)
        return 1.0f;
    if (x < b)
        return (x - a) / (b - a);
    return (d - x) / (d - c);
}

// ============================================================================
// FuzzyHumanSpeedController
// ============================================================================

FuzzyHumanSpeedController::FuzzyHumanSpeedController()
{
    // 规则结论映射：每条规则对应的输出中心索引
    // 0=STOP, 1=SLOW, 2=KEEP, 3=FULL
    //
    // R0: IF control=STOP                             → STOP
    // R1: IF control=SLOW AND offset=NEAR AND dist=CLOSE  → STOP   (很近→急刹)
    // R2: IF control=SLOW AND offset=NEAR AND dist=MEDIUM → SLOW   (中等→减速)
    // R3: IF control=SLOW AND offset=NEAR AND dist=FAR    → KEEP   (远→稳速)
    // R4: IF control=SLOW AND offset=MEDIUM               → SLOW
    // R5: IF control=SLOW AND offset=FAR                  → KEEP
    // R6: IF control=GO   AND offset=NEAR|MEDIUM          → KEEP
    // R7: IF control=FAR                                  → KEEP   (远区→谨慎)
    // R8: IF control=NONE                                 → FULL

    rules_out_[0] = 0; // → STOP
    rules_out_[1] = 0; // → STOP
    rules_out_[2] = 1; // → SLOW
    rules_out_[3] = 2; // → KEEP
    rules_out_[4] = 1; // → SLOW
    rules_out_[5] = 2; // → KEEP
    rules_out_[6] = 2; // → KEEP
    rules_out_[7] = 2; // → KEEP
    rules_out_[8] = 3; // → FULL
}

// ============================================================================
// 从 config 加载模糊输出中心值（icar 启动时自动调用）
// ============================================================================
void FuzzyHumanSpeedController::loadConfig(const Config &config)
{
    C_STOP = config.fuzzy_C_STOP;
    C_SLOW = config.fuzzy_C_SLOW;
    C_KEEP = config.fuzzy_C_KEEP;
    C_FULL = config.fuzzy_C_FULL;
}

// ============================================================================
// 简化接口（offset和distance用默认中性值）
// ============================================================================
float FuzzyHumanSpeedController::compute(int ctrl_state)
{
    return compute(ctrl_state, 1.0f, 0.5f);
}

// ============================================================================
// 核心接口：3 输入 Mamdani 模糊推理
// ============================================================================
float FuzzyHumanSpeedController::compute(int ctrl_state, float offset_ratio, float distance_ratio)
{
    float s = static_cast<float>(ctrl_state);
    float r = offset_ratio;
    float d = distance_ratio;

    // ---- Step 1: 模糊化 (Fuzzification) ----
    float ctrl_none = mu_ctrl_none(s);
    float ctrl_go = mu_ctrl_go(s);
    float ctrl_slow = mu_ctrl_slow(s);
    float ctrl_stop = mu_ctrl_stop(s);
    float ctrl_far = mu_ctrl_far(s);

    float off_near = mu_offset_near(r);
    float off_medium = mu_offset_medium(r);
    float off_far = mu_offset_far(r);

    float dist_close = mu_dist_close(d);
    float dist_medium = mu_dist_medium(d);
    float dist_far = mu_dist_far(d);

    // ---- Step 2: 规则评估 (Rule Evaluation, MIN for AND) ----

    float strength[NUM_RULES];

    // R0: IF control=STOP → STOP
    strength[0] = ctrl_stop;

    // R1: IF control=SLOW AND offset=NEAR AND dist=CLOSE → STOP
    strength[1] = std::min(std::min(ctrl_slow, off_near), dist_close);

    // R2: IF control=SLOW AND offset=NEAR AND dist=MEDIUM → SLOW
    strength[2] = std::min(std::min(ctrl_slow, off_near), dist_medium);

    // R3: IF control=SLOW AND offset=NEAR AND dist=FAR → KEEP
    strength[3] = std::min(std::min(ctrl_slow, off_near), dist_far);

    // R4: IF control=SLOW AND offset=MEDIUM → SLOW
    strength[4] = std::min(ctrl_slow, off_medium);

    // R5: IF control=SLOW AND offset=FAR → KEEP
    strength[5] = std::min(ctrl_slow, off_far);

    // R6: IF control=GO AND (offset=NEAR OR offset=MEDIUM) → KEEP
    strength[6] = std::min(ctrl_go, std::max(off_near, off_medium));

    // R7: IF control=FAR → KEEP
    strength[7] = ctrl_far;

    // R8: IF control=NONE → FULL
    strength[8] = ctrl_none;

    // ---- Step 3: 去模糊化 (Defuzzification, COG) ----
    return defuzzify_cog(strength, NUM_RULES);
}

// ============================================================================
// 去模糊化：重心法 (Center of Gravity)
//
// COG = Σ(strength_i × center_i) / Σ(strength_i)
//
// 这里使用 Mamdani 推理的简化：每条规则直接激活其结论模糊集的中心值，
// 权重为规则激活强度。对于单例输出模糊集，这等价于标准 COG。
// ============================================================================
float FuzzyHumanSpeedController::defuzzify_cog(const float *strengths, int n_rules) const
{
    float num = 0.0f;
    float den = 0.0f;

    const float centers[] = {C_STOP, C_SLOW, C_KEEP, C_FULL};

    for (int i = 0; i < n_rules; ++i)
    {
        float w = strengths[i];
        if (w > 0.0f)
        {
            int idx = rules_out_[i];
            num += w * centers[idx];
            den += w;
        }
    }

    if (den < 1e-8f)
        return C_FULL; // 无规则激活 → 全速

    return num / den;
}

// ============================================================================
// 输入隶属度函数实现
// ============================================================================

// ---- ControlState 模糊化：相邻状态有重叠保证过渡平滑 ----
//
// NONE: 三角(-0.5, 0.0, 0.9)
// GO:   三角( 0.1, 1.0, 1.9)
// SLOW: 三角( 1.1, 2.0, 2.9)
// STOP: 三角( 2.1, 3.0, 3.5)

float FuzzyHumanSpeedController::mu_ctrl_none(float s) const
{
    return tri_mf(s, -0.5f, 0.0f, 0.9f);
}

float FuzzyHumanSpeedController::mu_ctrl_go(float s) const
{
    return tri_mf(s, 0.1f, 1.0f, 1.9f);
}

float FuzzyHumanSpeedController::mu_ctrl_slow(float s) const
{
    return tri_mf(s, 1.1f, 2.0f, 2.9f);
}

float FuzzyHumanSpeedController::mu_ctrl_stop(float s) const
{
    return tri_mf(s, 2.1f, 3.0f, 3.9f);
}

// FAR: 三角(3.1, 4.0, 4.5)
float FuzzyHumanSpeedController::mu_ctrl_far(float s) const
{
    return tri_mf(s, 3.1f, 4.0f, 4.5f);
}

// ---- offset_ratio 模糊化 ----
//
// NEAR:  左肩三角 (0.0, 0.0, 1.00)  → 近区拉长
// MEDIUM: 三角   (0.5, 1.0, 2.0)   → 过渡区拉长
// FAR:   右肩梯形 (2.0, 3.0, 99, 99) → 远区缩短

float FuzzyHumanSpeedController::mu_offset_near(float r) const
{
    return tri_mf(r, 0.0f, 0.0f, 1.2f);   // 右端从 1.00 → 1.05，近区略扩大
}

float FuzzyHumanSpeedController::mu_offset_medium(float r) const
{
    return tri_mf(r, 0.55f, 1.1f, 2.10f);  // 左端 0.50→0.45，右端 2.00→2.10，过渡区略扩大
}

float FuzzyHumanSpeedController::mu_offset_far(float r) const
{
    return trap_mf(r, 2.10f, 3.10f, 100.0f, 100.0f);  // 起点 2.00→2.10，满值点 3.00→3.10，远区略缩小
}

// ---- distance_ratio ----
// distance_ratio = human_dist_y / IMAGE_H, 值越大行人越远
// CLOSE:  (0.0, 0.0, 0.15)  → 近区拉长
// MEDIUM: (0.06, 0.20, 0.20) → 过渡区拉长
// FAR:    (0.20, 1.0, 1.0)  → 远区缩短（起始点右移）

float FuzzyHumanSpeedController::mu_dist_close(float d) const
{
    return tri_mf(d, 0.0f, 0.0f, 0.15f);
}

float FuzzyHumanSpeedController::mu_dist_medium(float d) const
{
    return tri_mf(d, 0.06f, 0.20f, 0.20f);
}

float FuzzyHumanSpeedController::mu_dist_far(float d) const
{
    return tri_mf(d, 0.20f, 1.0f, 1.0f);
}

// ============================================================================
// output MFs
// ============================================================================

float FuzzyHumanSpeedController::mu_scale_stop(float x) const
{
    return tri_mf(x, 0.0f, 0.0f, 0.2f);
}

float FuzzyHumanSpeedController::mu_scale_slow(float x) const
{
    return tri_mf(x, 0.15f, 0.35f, 0.55f);
}

float FuzzyHumanSpeedController::mu_scale_keep(float x) const
{
    return tri_mf(x, 0.45f, 0.65f, 0.85f);
}

float FuzzyHumanSpeedController::mu_scale_full(float x) const
{
    return tri_mf(x, 0.8f, 1.0f, 1.0f);
}