#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include "../common/utils.hpp"
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <string.h>

class imgProcess
{
public:
    // ´«Í³cvÍ¼Ïñ´¦Àí
    void Img_process_CV(const cv::Mat &src, cv::Mat &morph_mask);

    // aiÓïÒå·Ö¸îÍ¼Ïñ´¦Àí
    void Img_process_AI(const cv::Mat &src, cv::Mat &morph_mask);
};
