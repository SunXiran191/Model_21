#include "imgProcess.hpp"
#include <algorithm>
#include <iostream>

using namespace std;
using namespace cv;

// 静态缓存 kernel，避免每帧重复创建
static const Mat kKernel3x3 = getStructuringElement(MORPH_RECT, Size(3, 3));
static const Mat kKernel5x5 = getStructuringElement(MORPH_RECT, Size(5, 5));

void imgProcess::Img_process_AI(const cv::Mat &src, cv::Mat &morph_mask)
{
    if (src.empty())
    {
        morph_mask.release();
        return;
    }

    // 掩码预处理
    if (src.channels() == 1)
    {
        src.copyTo(morph_mask);
    }
    else
    {
        cvtColor(src, morph_mask, COLOR_BGR2GRAY);
    }
    threshold(morph_mask, morph_mask, 0, 255, THRESH_BINARY);

    morphologyEx(morph_mask, morph_mask, MORPH_OPEN, kKernel5x5);
    morphologyEx(morph_mask, morph_mask, MORPH_CLOSE, kKernel5x5);
}

void imgProcess::Img_process_CV(const cv::Mat &src, cv::Mat &morph_mask)
{
    if (src.empty())
    {
        morph_mask.release();
        return;
    }

    // 掩码预处理
    if (src.channels() == 1)
    {
        src.copyTo(morph_mask);
    }
    else
    {
        cvtColor(src, morph_mask, COLOR_BGR2GRAY);
    }
    threshold(morph_mask, morph_mask, 0, 255, THRESH_BINARY);

    morphologyEx(morph_mask, morph_mask, MORPH_OPEN, kKernel3x3);
    morphologyEx(morph_mask, morph_mask, MORPH_CLOSE, kKernel3x3);
}