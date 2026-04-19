#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include "LineTracker.hpp"
#include "utils.hpp"
#include "general.h"

using namespace std;
using namespace cv;

#define PI 3.141592653589793
#define ROWSIMAGE 640         // 图像高度
#define COLSIMAGE 480         // 图像宽度
#define SAMPLE_DIST 0.05      // 重采样距离（米）
#define pixel_per_meter 100.f // 像素每米（假设值）
#define car_length 0.2        // 车长（米）
#define aim_distance_f 2.0f   // 远瞄准距离（米）

int aim_dis_pix = 600;

using POINT = cv::Point;

General general;

// 全局变量
vector<POINT> t_CenterEdge;
int t_CenterEdge_size = 0;
int CenterEdge_size = 0;
int aim_index_far;

// 重采样函数（基于距离）
void resample_points(const vector<POINT> &input, int input_size, vector<POINT> &output, int &output_size, float dist_threshold)
{
    output.clear();
    if (input_size < 2)
        return;

    output.push_back(input[0]);
    float accumulated_dist = 0.0f;

    for (int i = 1; i < input_size; ++i)
    {
        float dx = input[i].x - input[i - 1].x;
        float dy = input[i].y - input[i - 1].y;
        accumulated_dist += sqrt(dx * dx + dy * dy);

        if (accumulated_dist >= dist_threshold * pixel_per_meter)
        {
            output.push_back(input[i]);
            accumulated_dist = 0.0f;
        }
    }
    output_size = output.size();
}

int main()
{
    /********************************视频流读取测试********************************/
    std::string streamUrl = "udp://@:1234";

    cv::VideoCapture cap(streamUrl, cv::CAP_FFMPEG);
    if (!cap.isOpened())
    {
        std::cerr << "无法打开视频流: " << streamUrl << std::endl;
        return -1;
    }
    /********************************视频流读取测试********************************/

    LineTracker tracker;
    cv::Mat frame, mask, out_img;

    while (true)
    {
        if (!cap.read(frame) || frame.empty())
        {
            std::cerr << "读取视频帧失败 或 流结束" << std::endl;
            break;
        }

        out_img = frame.clone();
        std::vector<cv::Point> lane_points = tracker.ExtractArrows(frame, mask);

        if (lane_points.size() > 3)
        {
            std::vector<cv::Point> filtered_line =
                tracker.FitTrajectory_Poly((int)lane_points.size(), lane_points, frame);

            for (size_t i = 0; i + 1 < lane_points.size(); ++i)
            {
                cv::circle(out_img, lane_points[i], 4, cv::Scalar(0, 255, 255), -1);
            }
            for (size_t i = 0; i + 1 < filtered_line.size(); ++i)
            {
                cv::line(out_img, filtered_line[i], filtered_line[i + 1], cv::Scalar(255, 0, 0), 3);
            }

            // 初始化 t_CenterEdge 从 lane_points
            t_CenterEdge.clear();
            for (const auto &pt : filtered_line)
            {
                t_CenterEdge.emplace_back(pt.x, pt.y);
            }
            t_CenterEdge_size = t_CenterEdge.size();

            float min_dist = 10000000;
            int begin_id = -1;
            bool center_effective_flag = false; // 中线有效标志

            float cx = COLSIMAGE / 2.0f; // 车轮对应点 (纯跟踪起始点)
            float cy = ROWSIMAGE * 0.95f;

            // 找最近点(起始点中线归一化)
            for (int i = 0; i < t_CenterEdge_size; i++)
            {
                float dx = t_CenterEdge[i].x - cx;
                float dy = t_CenterEdge[i].y - cy;
                float dist = sqrt(dx * dx + dy * dy);
                if (dist < min_dist)
                {
                    min_dist = dist;
                    begin_id = i;
                }
            }

            begin_id = clamp(begin_id, 0, t_CenterEdge_size - 1);

            std::vector<POINT> temp_center;

            int temp_center_size;
            printf("begin id %d size %d\n", begin_id, t_CenterEdge_size);
            if ((begin_id >= 0 && t_CenterEdge_size - begin_id >= 3))
            {
                center_effective_flag = true;
                if (begin_id > 0)
                {
                    cx = t_CenterEdge[begin_id].x;
                    cy = t_CenterEdge[begin_id].y;
                }
                for (int i = begin_id; i < t_CenterEdge_size; i++)
                {
                    temp_center.emplace_back(t_CenterEdge[i].x, t_CenterEdge[i].y);
                }

                temp_center_size = temp_center.size();
                t_CenterEdge.clear();
                t_CenterEdge_size = 0;
                resample_points(temp_center, temp_center_size, t_CenterEdge,
                                t_CenterEdge_size, SAMPLE_DIST * pixel_per_meter);

                double min_dis = 1000000;

                for (int i = 1; i < t_CenterEdge_size; i++)
                {
                    double dx = t_CenterEdge[i].x - cx;
                    double dy = cy - t_CenterEdge[i].y;
                    double dn = sqrt(dx * dx + dy * dy);

                    double dis = aim_distance_f * pixel_per_meter - dn;
                    if (dis < 0)
                    {
                        dis *= -1;
                    }
                    if (dis < min_dis)
                    {
                        aim_index_far = i;
                        min_dis = dis;
                    }
                }

                float dx = t_CenterEdge[aim_index_far].x - cx;                                // rptsn[aim_idx__far][0] - cx;
                float dy = cy - t_CenterEdge[aim_index_far].y + car_length * pixel_per_meter; // cy - rptsn[aim_idx__far][1];
                float dn = sqrt(dx * dx + dy * dy);
                float error = (-atanf(pixel_per_meter * 2 * car_length * dx / dn / dn) * 180 / PI);
                assert(!isnan(error));
                printf("far_dx:%f,far_dy %f\n", dx, dy);
                printf("cx %f cy %f\n", cx, cy);
                printf("error_far: %f degrees\n", error); // 添加打印偏差角
            }
        }

        cv::imshow("巡线结果", out_img);

        int key = cv::waitKey(1);
        if (key == 27 || key == 'q')
            break;
    }

    cap.release();
    return 0;
}
