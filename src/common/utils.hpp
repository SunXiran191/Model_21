#ifndef UTILSFORALL_H
#define UTILSFORALL_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <numeric>

int clip(int x, int low, int up);
float clipf(float x, float low, float up);
int factorial(int x);
void swap(int *a, int *b);

#define max(a, b) (((a) > (b)) ? (a) : (b))
#define min(a, b) (((a) < (b)) ? (a) : (b))
#define GET_PIX_1C(IMG, H, W) (IMG[(H) * IMAGE_W + (W)])                // 灰度图获取坐标灰度值
#define GET_PIX_3C(IMG, H, W, C) (IMG[((H) * IMAGE_W + (W)) * 3 + (C)]) // rgb图获取坐标三通道值

#define PI 3.141593

#define IMAGE_H (480) // 图像高度（依据摄像头不同可能有变化
#define IMAGE_W (640) // 图像宽度（依据摄像头不同可能有变化

#define WHITE (255)
#define BLACK (0)

#define BEGINH_L (100) // 左起始点扫线开始y坐标  可能需要调整
#define BEGINH_R (100) // 右起始点扫线开始y坐标  可能需要调整

#define SELFADAPT_KERNELSIZE (7)
#define FILTER_KERNELSIZE (7)
#define PIXPERMETER (200)    // 逆透视时设置的像素比例（图像上N个像素为实际的1m）
#define RESAMPLEDIST (0.005) // 重采样间隔距离（单位：米）
#define ANGLEDIST (0.03)
#define ROADWIDTH (0.02) // 道路宽度
#define AIMDISTANCE (0.03)
#define CORRECTDIST_H (0) // 逆透视变换后h平移距离(像素
#define CORRECTDIST_W (0) // 逆透视变换后w平移距离(像素
#define TURN_THRESHOLD (12.0)

#define PREDICT_THRESH (10)
#define LOST_THRESH (10)

#endif /* UTILSFORALL_H */
