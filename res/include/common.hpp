#pragma once

/*
 * @file common common.hpp
 * @author Leo
 * @brief 通用方法类
 * @version 0.1
 * @date 2024-01-12
 *
 * @copyright Copyright (c) 2024
 */

#include "json.hpp"
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <opencv2/highgui.hpp> //OpenCV终端部署
#include <opencv2/opencv.hpp>  //OpenCV终端部署

using namespace std;
using namespace cv;

#define POINTS_MAX_LEN 180

#define COLSIMAGE 640    // 图像的列数
#define ROWSIMAGE 480    // 图像的行数
#define COLSIMAGEIPM 320 // IPM图像的列数
#define ROWSIMAGEIPM 400 // IPM图像的行数
#define PWMSERVOMAX 4700 // 舵机PWM最大值（左）1840
#define PWMSERVOMID 3840 // 舵机PWM中值 1520
#define PWMSERVOMIN 3000 // 舵机PWM最小值（右）1200
#define PI (3.1415926535898)

#define PIXPERMETER 222.222 // 一米在画面中所占的像素数

/**
 * @brief 场景类型（路况）
 *
 */
// enum Scene_status
// {
//     NormalScene = 0, // 基础赛道
//     CrossScene,      // 十字道路
//     RingScene,       // 环岛道路
//     BridgeScene,     // 坡道区
//     ObstacleScene,   // 障碍区
//     CateringScene,   // 快餐店
//     LaybyScene,      // 临时停车区
//     ParkingScene,    // 停车区
//     StopScene        // 停车（结束）
// };

/**
 * @brief 构建二维坐标
 *
 */
struct POINT
{
    int x = 0;
    int y = 0;
    float slope = 0.0f;
    float angle = 0.0f;

    POINT() {};
    POINT(int x, int y) : x(x), y(y) {};
};

/**
 * @brief UI综合图像绘制
 *
 */
class Display
{
private:
    bool enable = false; // 显示窗口使能
    int sizeWindow = 1;  // 窗口数量
    cv::Mat imgShow;     // 窗口图像
public:
    /**
     * @brief 显示窗口初始化
     *
     * @param size 窗口数量(1~7)
     */
    Display(const int size)
    {
        if (size <= 0 || size > 7)
            return;

        // cv::namedWindow("ICAR", WINDOW_NORMAL);                // 图像名称
        // cv::resizeWindow("ICAR", COLSIMAGE * size, ROWSIMAGE); // 分辨率

        imgShow = cv::Mat::zeros(ROWSIMAGE, COLSIMAGE * size, CV_8UC3);
        enable = true;
        sizeWindow = size;
    };

    /**
     * @brief 设置新窗口属性
     *
     * @param index 窗口序号
     * @param name 窗口名称
     * @param img 显示图像
     */
    void setNewWindow(int index, string name, Mat img)
    {
        // 数据溢出保护
        if (!enable || index <= 0 || index > sizeWindow)
            return;

        if (img.cols <= 0 || img.rows <= 0)
            return;

        Mat imgDraw = img.clone();

        if (imgDraw.type() == CV_8UC1) // 非RGB类型的图像
            cvtColor(imgDraw, imgDraw, cv::COLOR_GRAY2BGR);

        // 图像缩放
        if (imgDraw.cols != COLSIMAGE || imgDraw.rows != ROWSIMAGE)
        {
            float fx = COLSIMAGE / imgDraw.cols;
            float fy = ROWSIMAGE / imgDraw.rows;
            if (fx <= fy)
                resize(imgDraw, imgDraw, Size(COLSIMAGE, ROWSIMAGE), fx, fx);
            else
                resize(imgDraw, imgDraw, Size(COLSIMAGE, ROWSIMAGE), fy, fy);
        }

        // 限制图片标题长度
        string text = "[" + to_string(index) + "] ";
        if (name.length() > 15)
            text = text + name.substr(0, 15);
        else
            text = text + name;

        putText(imgDraw, text, Point(10, 20), cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(255, 0, 0), 0.5);

        Rect placeImg = cvRect(COLSIMAGE * (index - 1), 0, COLSIMAGE, ROWSIMAGE);
        imgDraw.copyTo(imgShow(placeImg));
    }

    /**
     * @brief 融合后的图像显示
     *
     */
    void show(void)
    {
        if (enable)
            imshow("ICAR", imgShow);
    }
};
