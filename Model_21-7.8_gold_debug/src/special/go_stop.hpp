#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include "../../res/configs/param.hpp"
#include "../../res/include/aiget.hpp"
#include "../common/utils.hpp"
#include "objects.hpp"

class GoStop
{
public:
    GoStop();
    GoStop(Config &config);

    // 将 STOP/GO 检测框绘制到语义分割掩码上（填充为道路区域），使巡线能穿过标志牌
    // segmask: CV_8UC1 掩码（会被原地修改），255=道路，0=非道路
    static void paintBoxesToSegMask(
        cv::Mat &segmask,
        const std::vector<PredictResult> &stop_results,
        const std::vector<PredictResult> &go_results);

    // 主入口：处理 STOP/GO 标志牌的轨迹修正
    // stop_point/go_point: IPM 变换后的标志中心坐标
    // trajectory: 等距采样后的 IPM 轨迹点（会被原地修改）
    void processGoStop(
        const std::vector<cv::Point> &stop_point,
        const std::vector<cv::Point> &go_point,
        std::vector<cv::Point> &trajectory,
        int &trajectory_size);

    // ====== 配置参数 ======
    int go_stop_dist_forward;  // 沿斜率向上延长距离（IPM像素）
    int go_stop_dist_backward; // 沿斜率向下延长距离（IPM像素）
    float confidence_threshold;

private:
    // 对 STOP/GO 标志中心点做直线拟合，返回方向向量（指向图像上方）
    cv::Point2f fitLineThroughCenters(const std::vector<cv::Point> &centers) const;
};
