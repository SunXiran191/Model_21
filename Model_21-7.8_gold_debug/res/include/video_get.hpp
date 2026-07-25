#ifndef SHM_VIDEO_READER_H
#define SHM_VIDEO_READER_H

#include <string>

#include <opencv2/opencv.hpp>

// 配置
#define SHM_NAME "shm_ar_video"
#define SHM_HEADER_SIZE 16

// 初始化连接共享内存
bool init_shm(const std::string &shm_name = SHM_NAME, size_t map_bytes = 10 * 1024 * 1024);

// 获取最新一帧视频
cv::Mat get_realtime_frame(const std::string &shm_name = SHM_NAME, size_t map_bytes = 10 * 1024 * 1024);

// 关闭共享内存
void close_shm();

#endif