#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <numeric>

int clip(int x, int low, int up);
float clipf(float x, float low, float up);
int factorial(int x);
void swap(int *a, int *b);

// 定义 class_id映射关系（顺序与 labels.txt 严格一致）
// static const int CLASS_ID_STOP = 0;
// static const int CLASS_ID_BRANCHSIGN = 1;
// static const int CLASS_ID_SPEEDSIGN = 2;
// static const int CLASS_ID_LIGHT_GREEN = 3;
// static const int CLASS_ID_LIGHT_RED = 4;
// static const int CLASS_ID_LIGHT_YELLOW = 5;
// static const int CLASS_ID_ZEBRALINE = 6;
// static const int CLASS_ID_GOLD = 7;
// static const int CLASS_ID_HUMAN = 8;
// static const int CLASS_ID_CAR = 9;

static const int CLASS_ID_STOP = 0;
static const int CLASS_ID_GO = 1;
static const int CLASS_ID_BRANCHSIGN = 2;
static const int CLASS_ID_SPEEDSIGN = 3;
static const int CLASS_ID_LIGHT_GREEN = 4;
static const int CLASS_ID_LIGHT_RED = 5;
static const int CLASS_ID_LIGHT_YELLOW = 6;
static const int CLASS_ID_ZEBRALINE = 7;
static const int CLASS_ID_GOLD = 8;
static const int CLASS_ID_HUMAN = 9;
static const int CLASS_ID_CAR = 10;

#define PI (3.1415926536)

#define GET_PIX_1C(IMG, H, W) (IMG[(H) * IMAGE_W + (W)])                // 灰度图获取坐标灰度值
#define GET_PIX_3C(IMG, H, W, C) (IMG[((H) * IMAGE_W + (W)) * 3 + (C)]) // rgb图获取坐标三通道值

#define IMAGE_H (480) // 图像高度（依据摄像头不同可能有变化
#define IMAGE_W (640) // 图像宽度（依据摄像头不同可能有变化

#define WHITE (255)
#define BLACK (0)

#define BEGINH_L (100) // 左起始点扫线开始y坐标  可能需要调整
#define BEGINH_R (100) // 右起始点扫线开始y坐标  可能需要调整

#define PTS_LINE_NUM (50)
#define SELFADAPT_KERNELSIZE (7)
#define FILTER_KERNELSIZE (7)
#define PIXPERMETER (450)   // 逆透视时设置的像素比例（图像上N个像素为实际的1m）
#define RESAMPLEDIST (0.02) // 重采样间隔距离（单位：米）
#define ANGLEDIST (0.03)
#define ROADWIDTH (0.02)    // 道路宽度
#define ROADWIDTH_PIX (220) // 道路宽度_像素
#define AIMDISTANCE (0.03)
#define CORRECTDIST_H (0) // 逆透视变换后h平移距离(像素
#define CORRECTDIST_W (0) // 逆透视变换后w平移距离(像素
#define TURN_THRESHOLD (12.0)

#define LOST_THRESH (10)
