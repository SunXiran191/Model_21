#include "video_get.hpp"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <iostream>

#define SHM_NAME "shm_ar_video"
#define SHM_HEADER_SIZE 16

static int shm_fd = -1;
static void *shm_ptr = nullptr;
static uint64_t last_fid = 0;
static std::string current_shm_name = SHM_NAME;
static size_t current_map_bytes = 10 * 1024 * 1024;

bool init_shm(const std::string &shm_name, size_t map_bytes)
{
    current_shm_name = shm_name;
    current_map_bytes = map_bytes;

    shm_fd = shm_open(current_shm_name.c_str(), O_RDONLY, 0666);
    if (shm_fd == -1)
    {
        return false;
    }
    shm_ptr = mmap(NULL, current_map_bytes, PROT_READ, MAP_SHARED, shm_fd, 0);
    if (shm_ptr == MAP_FAILED)
    {
        close(shm_fd);
        shm_fd = -1;
        shm_ptr = nullptr;
        return false;
    }
    return true;
}

void close_shm()
{
    if (shm_ptr)
        munmap(shm_ptr, current_map_bytes);
    if (shm_fd != -1)
        close(shm_fd);
    shm_ptr = nullptr;
    shm_fd = -1;
}

// ===================== ���ռ��棺�޿��� + ��ȫ + ���١� =====================
cv::Mat get_realtime_frame(const std::string &shm_name, size_t map_bytes)
{
    if (!shm_ptr || current_shm_name != shm_name || current_map_bytes != map_bytes)
    {
        close_shm();
        if (!init_shm(shm_name, map_bytes))
        {
            return {};
        }
    }

    try
    {
        uint8_t *buf = (uint8_t *)shm_ptr;
        uint64_t fid = *(uint64_t *)buf;
        uint32_t w = *(uint32_t *)(buf + 8);
        uint32_t h = *(uint32_t *)(buf + 12);

        if (fid == last_fid)
        {
            return {};
        }
        last_fid = fid;

        const uint8_t *data = buf + SHM_HEADER_SIZE;
        cv::Mat bgr(h, w, CV_8UC3);

        // SHM 图像 RGB→BGR 转换
        for (uint32_t y = 0; y < h; ++y)
        {
            const uint8_t *src_row = data + static_cast<size_t>(y) * static_cast<size_t>(w) * 3;
            uint8_t *dst_row = bgr.ptr<uint8_t>(static_cast<int>(y));
            for (uint32_t x = 0; x < w; ++x)
            {
                const size_t src_idx = static_cast<size_t>(x) * 3;
                dst_row[src_idx + 0] = src_row[src_idx + 2];
                dst_row[src_idx + 1] = src_row[src_idx + 1];
                dst_row[src_idx + 2] = src_row[src_idx + 0];
            }
        }
        return bgr;
    }
    catch (...)
    {
        close_shm();
        return {};
    }
}