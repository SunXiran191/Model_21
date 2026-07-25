#include "thread.hpp"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>
#include "../include/sendUDP.hpp"

#include "../inference/ocr/ocr_api.hpp" // BaiduApi
#include "../special/branch.hpp"        // Branch::BranchState
using namespace cv;
using namespace std;

// ============================================================
//  线程绑核辅助宏 — 将当前线程固定到指定 CPU 核心
// ============================================================
#define PIN_TO_CORE(core)                                          \
    do                                                             \
    {                                                              \
        cpu_set_t cpu;                                             \
        CPU_ZERO(&cpu);                                            \
        CPU_SET(core, &cpu);                                       \
        pthread_setaffinity_np(pthread_self(), sizeof(cpu), &cpu); \
    } while (0)

// ARM64 低延迟自旋等待 — 替代 usleep，消除 ~1-4ms 内核唤醒抖动
// 仅用于独占大核 (A76) 的关键线程；小核非实时线程继续使用 usleep 以省电
// 混合策略: 先自旋，无数据则 usleep(0) 让出 CPU
//   每线程使用不同自旋上限（基于 TLS 地址），彻底防止多线程同步让出导致流水线断流
#define SPIN_HINT()                                                               \
    do                                                                            \
    {                                                                             \
        static __thread int _spin_cnt = 0;                                        \
        static __thread int _spin_max =                                           \
            800 + ((int)(uintptr_t)&_spin_cnt % 1200); /* 800~1999, 各线程不同 */ \
        if (++_spin_cnt > _spin_max)                                              \
        {                                                                         \
            usleep(0); /* 让出 CPU 一个调度滴答 */                                \
            _spin_cnt = 0;                                                        \
        }                                                                         \
        else                                                                      \
        {                                                                         \
            __asm__ volatile("yield" ::: "memory");                               \
        }                                                                         \
    } while (0)

extern std::mutex mtx_produce;
// seg_mask 序列号：infer_seg_thread 每写入新 mask 后递增，run_thread 借此判断 mask 是否过期
vector<Point> draw_line;
cv::Point g_aim_point;
std::mutex mtx_draw;
static std::mutex mtx_frame;
static cv::Mat latest_frame;
static std::atomic<uint64_t> g_frame_seq(0);

// ===== LATENCY_MEASURE: frame_seq → timestamp ring buffer =====
// produce_thread 写入采集时间戳，run_thread 按 frame_seq 精确查找
// 删除方法：grep LATENCY_MEASURE 删除所有标记块即可
namespace latency
{
    static constexpr size_t kRingSize = 64;
    struct Entry
    {
        uint64_t frame_seq = 0;
        uint64_t timestamp_us = 0;
    };
    static Entry ring[kRingSize];

    inline void push(uint64_t seq, uint64_t ts_us)
    {
        ring[seq % kRingSize] = {seq, ts_us};
    }
    inline uint64_t lookup(uint64_t seq)
    {
        const Entry &e = ring[seq % kRingSize];
        return (e.frame_seq == seq) ? e.timestamp_us : 0;
    }
} // namespace latency
// ===== LATENCY_MEASURE_END =====

// ============================================================
//  流线程独立缓冲区（避免与 mtx_produce 竞争）
// ============================================================
struct StreamSnap
{
    cv::Mat img;                            // 深拷贝后的渲染缓冲
    cv::Mat morph_mask;                     // 浅拷贝
    std::vector<PredictResult> predictions; // 深拷贝
};
static StreamSnap g_stream_snap;
static std::mutex mtx_stream;
static std::atomic<uint64_t> g_stream_seq{0};

// static const useconds_t PREDICT_IDLE_US = 10000;
static const int MJPEG_PORT = 8083;
static const int MJPEG_QUALITY = 10;
static const useconds_t STREAM_IDLE_US = 3000;

namespace
{
    static const int kSegShmHeaderSize = 16;
    static const useconds_t kSegShmIdleUs = 2000;

    class SegShmReader
    {
    public:
        SegShmReader(const std::string &name, size_t map_bytes)
            : name_(name), map_bytes_(map_bytes), fd_(-1), ptr_(MAP_FAILED), last_fid_(0) {}

        ~SegShmReader() { close(); }

        bool open()
        {
            std::string path = "/dev/shm/" + name_;
            fd_ = ::open(path.c_str(), O_RDONLY);
            if (fd_ < 0)
            {
                std::cerr << "[SEG_SHM] open failed: " << path << std::endl;
                return false;
            }
            ptr_ = mmap(NULL, map_bytes_, PROT_READ, MAP_SHARED, fd_, 0);
            if (ptr_ == MAP_FAILED)
            {
                std::cerr << "[SEG_SHM] mmap failed" << std::endl;
                ::close(fd_);
                fd_ = -1;
                return false;
            }
            return true;
        }

        void close()
        {
            if (ptr_ != MAP_FAILED)
            {
                munmap(ptr_, map_bytes_);
                ptr_ = MAP_FAILED;
            }
            if (fd_ >= 0)
            {
                ::close(fd_);
                fd_ = -1;
            }
        }

        // Returns true if new mask was read.
        bool read_mask(cv::Mat &mask)
        {
            if (ptr_ == MAP_FAILED)
            {
                if (!open())
                    return false;
            }

            const unsigned char *buf = static_cast<const unsigned char *>(ptr_);

            uint64_t fid = 0;
            uint32_t w = 0, h = 0;
            std::memcpy(&fid, buf, 8);
            std::memcpy(&w, buf + 8, 4);
            std::memcpy(&h, buf + 12, 4);

            if (fid == last_fid_ || w == 0 || h == 0)
                return false;

            last_fid_ = fid;

            size_t data_len = static_cast<size_t>(w) * h;
            if (kSegShmHeaderSize + data_len > map_bytes_)
                return false;

            mask = cv::Mat(static_cast<int>(h), static_cast<int>(w), CV_8UC1,
                           const_cast<unsigned char *>(buf + kSegShmHeaderSize))
                       .clone();
            return true;
        }

    private:
        std::string name_;
        size_t map_bytes_;
        int fd_;
        void *ptr_;
        uint64_t last_fid_;
    };
} // namespace

static cv::Scalar getBoxColor(int class_id)

{

    switch (class_id)

    {

    case 0:
        return cv::Scalar(0, 215, 255);
    case 1:
        return cv::Scalar(0, 0, 255);
    case 2:
        return cv::Scalar(0, 255, 0);
    case 3:
        return cv::Scalar(255, 0, 0);
    case 4:
        return cv::Scalar(0, 0, 0);
    case 5:
        return cv::Scalar(255, 255, 0);
    case 6:
        return cv::Scalar(255, 0, 255);
    case 7:
        return cv::Scalar(0, 255, 255);
    case 8:
        return cv::Scalar(180, 150, 0);
    case 9:
        return cv::Scalar(20, 150, 150);
    case 10:
        return cv::Scalar(150, 250, 150);

    default:
        return cv::Scalar(255, 255, 255);
    }
}

void run_thread(Produce &G_produce, const Config &config)
{
    PIN_TO_CORE(7); // 控制闭环最关键，独占 Core 7 (A76 大核)

    // 控制线程：消费推理线程写入的结果，执行巡线/避障决策并下发电机控制。
    Uart uart0("/dev/ttyUSB0");
    Uart uart1("/dev/ttyUSB1");
    Uart *uart = &uart0;
    // UdpTelemetrySender udp_sender("192.168.137.1", 9007);
    bool uart_ready = (uart0.open() == 0);
    if (!uart_ready)
    {
        uart = &uart1;
        uart_ready = (uart1.open() == 0);
    }

    if (!uart_ready)
    {
        std::cerr << "warning: uart open failed, control output disabled" << std::endl;
    }
    Standard standard(config);
    uint32_t run_frame_count = 0;
    auto run_last_tick = std::chrono::steady_clock::now();
    uint64_t last_produce_seq = 0;
    uint64_t last_seg_seq = 0;
    float global_speed = 1.0f;
    double global_fps = 0.f;
    LostLine lostline;
    const int kUartRxTimeoutMs = 2;
    const double kUartLogIntervalSec = 0.5;
    auto uart_log_tick = std::chrono::steady_clock::now();
    uint32_t uart_no_frame_count = 0;
    uint64_t prof_frames = 0;
    double prof_run_ms = 0.0;
    double prof_uart_ms = 0.0;
    double prof_total_ms = 0.0;
    auto prof_last_tick = std::chrono::steady_clock::now();
    while (!g_exit)
    {
        const uint64_t current_produce_seq = g_produce_seq.load(std::memory_order_acquire);
        if (current_produce_seq == last_produce_seq)
        {
            SPIN_HINT();
            continue;
        }
        // const uint64_t current_produce_seq = g_seg_seq.load(std::memory_order_acquire);
        // if (current_produce_seq == last_seg_seq)
        // {
        //     usleep(500);
        //     continue;
        // }
        last_produce_seq = current_produce_seq;
        global_fps = g_run_thread_fps.load();
        // if (global_speed != 0.0f)
        // {
        //     global_speed = speed_now.load();
        // }
        auto loop_start = std::chrono::steady_clock::now();
        auto run_start = loop_start;
        // ===== LATENCY_MEASURE: snapshot frame_seq before processing =====
        const uint64_t processed_frame_seq = G_produce.frame_seq;
        // ===== LATENCY_MEASURE_END =====
        TaskData dst = standard.run(G_produce, global_speed, global_fps);
        auto run_end = std::chrono::steady_clock::now();
        // ===== LATENCY_MEASURE: compute e2e latency (produce → run output) =====
        {
            uint64_t cap_ts = latency::lookup(processed_frame_seq);
            if (cap_ts != 0)
            {
                uint64_t now_us = static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::microseconds>(
                        run_end.time_since_epoch())
                        .count());
                uint64_t lat_us = now_us - cap_ts;
                static uint64_t lat_acc = 0;
                static int lat_cnt = 0;
                lat_acc += lat_us;
                // if (++lat_cnt >= 30) {
                //     printf("[latency] e2e=%.1fms (produce→run, avg over %d frames)\n",
                //            (static_cast<double>(lat_acc) / lat_cnt) / 1000.0, lat_cnt);
                //     lat_acc = 0;
                //     lat_cnt = 0;
                // }
            }
        }
        // ===== LATENCY_MEASURE_END =====
        prof_run_ms += std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(run_end - run_start).count();
        g_error.store(dst.error, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lock(mtx_draw);
            draw_line = (standard.trackstate == Standard::TRACK_AI_MIDDLE)
                            ? standard.s_t_trackPoints_AI
                            : standard.s_t_trackPoints_CV;
            g_aim_point = standard.aim_point;
        }

        // ---- 流缓冲区暂禁用 ----
        // ---- 流缓冲区暂禁用 ----

        ++run_frame_count;
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(now - run_last_tick).count();
        if (elapsed >= 1.0)
        {
            g_run_thread_fps.store(run_frame_count / elapsed, std::memory_order_relaxed);
            run_frame_count = 0;
            run_last_tick = now;
        }

        if (uart_ready)
        {
            auto uart_start = std::chrono::steady_clock::now();
            if (stop_flag)
            {
                dst.speed = 0.0f;
            }
            float temp_speed = 0.0f;
            const bool got_speed = 0; // uart->receiveSpeed(temp_speed, kUartRxTimeoutMs);
            if (got_speed)
            {
                speed_now.store(temp_speed);
            }
            else
            {
                ++uart_no_frame_count;
            }
            lostline.lost_line_update(dst.error, dst.x_error, dst.track_side, dst.error, dst.x_error);
            // 每 30 帧 (~1Hz) 打印一次速度，避免每帧 printf 阻塞 stdout 导致卡顿
            {
                static int log_cnt = 0;
                if (++log_cnt % 30 == 0)
                {
                    printf("speed:%.2f\n", dst.speed);
                    log_cnt = 0;
                }
            }
            uart->carControl(dst.speed, dst.error, dst.x_error);
            auto uart_log_now = std::chrono::steady_clock::now();
            double uart_log_elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(uart_log_now - uart_log_tick).count();
            if (uart_log_elapsed >= kUartLogIntervalSec)
            {
                printf("[ctrl] speed: %.2f, error: %.2f, x_error: %.2f\n",
                       dst.speed, dst.error, dst.x_error);
                uart_no_frame_count = 0;
                uart_log_tick = uart_log_now;
            }
            auto uart_end = std::chrono::steady_clock::now();
            prof_uart_ms += std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(uart_end - uart_start).count();
        }

        // UDP 遥测不依赖 UART，独立发送
        // udp_sender.send_data(dst.error, dst.x_error);

        auto loop_end = std::chrono::steady_clock::now();
        prof_total_ms += std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(loop_end - loop_start).count();
        ++prof_frames;

        double prof_elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(loop_end - prof_last_tick).count();
        if (prof_elapsed >= 1.0)
        {
            const double denom = prof_frames > 0 ? static_cast<double>(prof_frames) : 1.0;
            const double prof_fps = prof_frames / prof_elapsed;
            if (config.run_debug)
                printf("[profile run] fps=%.1f run=%.2fms uart=%.2fms total=%.2fms\n",
                       prof_fps,
                       prof_run_ms / denom,
                       prof_uart_ms / denom,
                       prof_total_ms / denom);
            prof_frames = 0;
            prof_run_ms = 0.0;
            prof_uart_ms = 0.0;
            prof_total_ms = 0.0;
            prof_last_tick = loop_end;
        }
    }
}

void produce_thread(Produce &G_produce, const Config &config)

{
    PIN_TO_CORE(6); // 摄像头采集绑定到 core 6

    // 采集线程
    uint32_t prod_frame_count = 0;
    auto prod_last_tick = std::chrono::steady_clock::now();
    // 预分配帧缓冲，用 swap 替代 clone 避免每帧 900KB 堆分配
    cv::Mat frame_buffer;
    while (!g_exit)
    {
        // cv::Mat out_img = get_realtime_frame("shm_ar_video", 10 * 1024 * 1024);
        cv::Mat out_img = get_realtime_frame("shm_ar_video", 3 * 640 * 480 + 16);
        // cv::flip(out_img, out_img, -1); // 垂直翻转，赛道在上方 !!!!!!!!!!!!!!!!!!!!
        if (out_img.empty())
        {
            SPIN_HINT();
            continue;
        }

        uint64_t new_frame_seq = 0;
        {
            std::lock_guard<std::mutex> lock(mtx_frame);
            // swap 交换数据指针 O(1)，out_img 取走旧缓冲（由 get_realtime_frame 下次复用）
            cv::swap(latest_frame, out_img);
            new_frame_seq = g_frame_seq.fetch_add(1, std::memory_order_release) + 1;
        }
        // ===== LATENCY_MEASURE =====
        latency::push(new_frame_seq, static_cast<uint64_t>(
                                         std::chrono::duration_cast<std::chrono::microseconds>(
                                             std::chrono::steady_clock::now().time_since_epoch())
                                             .count()));
        // ===== LATENCY_MEASURE_END =====

        ++prod_frame_count;
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(now - prod_last_tick).count();
        if (elapsed >= 1.0)
        {
            if (config.run_debug)
                printf("[profile produce] fps=%.1f\n", prod_frame_count / elapsed);
            prod_frame_count = 0;
            prod_last_tick = now;
        }
    }
}

void infer_det_thread(Produce &G_produce, const Config &config)
{
    PIN_TO_CORE(5); // NPU 前后处理（CPU 密集），绑定到 core 5

    // 目标检测推理线程
    PpyoloeDetector detector;
    PpyoloeConfig infer_config;

    infer_config.model_path = config.infer_model_path;
    infer_config.labels_path = config.infer_labels_path;
    infer_config.udp_ip = "127.0.0.1"; // config.infer_udp_ip;
    infer_config.udp_port = config.infer_udp_port;
    infer_config.enable_udp = config.infer_enable_udp;
    infer_config.confidence_threshold = config.confidence_threshold;
    infer_config.nms_threshold = config.infer_nms_threshold;
    infer_config.shm_name = "shm_ar_video";
    infer_config.shm_map_bytes = 3 * 640 * 480 + 16;
    infer_config.npu_core = 2;
    infer_config.verbose = config.infer_det_debug;

    if (!detector.load(infer_config))
    {
        std::cerr << "warning: ppyoloe detector load failed" << std::endl;
        g_exit = true;
        return;
    }

    uint64_t frame_seq = 0;
    uint64_t last_frame_seq = 0;
    uint64_t prof_frames = 0;
    double prof_infer_ms = 0.0;
    double prof_total_ms = 0.0;
    auto prof_last_tick = std::chrono::steady_clock::now();

    while (!g_exit)
    {
        const uint64_t current_frame_seq = g_frame_seq.load(std::memory_order_acquire);
        if (current_frame_seq == last_frame_seq)
        {
            SPIN_HINT();
            continue;
        }
        last_frame_seq = current_frame_seq;

        cv::Mat out_img;
        {
            std::lock_guard<std::mutex> lock(mtx_frame);
            out_img = latest_frame;
        }

        if (out_img.empty())
        {
            SPIN_HINT();
            continue;
        }

        std::vector<PredictResult> detections;
        auto loop_start = std::chrono::steady_clock::now();
        auto infer_start = loop_start;
        if (!detector.infer(out_img, detections))
        {
            if (config.infer_det_debug)
                std::cerr << "[infer_det] ppyoloe infer failed, skip this frame" << std::endl;
            SPIN_HINT();
            continue;
        }
        auto infer_end = std::chrono::steady_clock::now();
        prof_infer_ms += std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(infer_end - infer_start).count();

        {
            lock_guard<mutex> lock(mtx_produce);
            G_produce.predict_results = detections;
            G_produce.img = out_img;
            // ===== LATENCY_MEASURE =====
            G_produce.frame_seq = last_frame_seq;
            // ===== LATENCY_MEASURE_END =====
            g_produce_seq.fetch_add(1, std::memory_order_release);
        }

        ++frame_seq;
        auto loop_end = std::chrono::steady_clock::now();
        prof_total_ms += std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(loop_end - loop_start).count();
        ++prof_frames;

        double prof_elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(loop_end - prof_last_tick).count();
        if (prof_elapsed >= 1.0)
        {
            const double denom = prof_frames > 0 ? static_cast<double>(prof_frames) : 1.0;
            const double prof_fps = prof_frames / prof_elapsed;
            if (config.infer_det_debug)
                printf("[infer_det] fps=%.1f total=%.2fms\n",
                       prof_fps,
                       prof_total_ms / denom);
            prof_frames = 0;
            prof_infer_ms = 0.0;
            prof_total_ms = 0.0;
            prof_last_tick = loop_end;
        }
    }
}

void infer_seg_thread(Produce &G_produce, const Config &config)
{
    PIN_TO_CORE(3); // 分割轻量级，放小核 core 3
    // 分割推理线程：通过共享内存接收 Python 推理输出的 mask。
    // const bool enable_seg = config.enAI && config.infer_seg_enable_shm;
    const bool enable_seg = true;
    if (!enable_seg)
    {
        {
            lock_guard<mutex> lock(mtx_produce);
            G_produce.seg_mask.release();
        }
        while (!g_exit)
        {
            usleep(10000);
        }
        return;
    }

    SegShmReader reader("shm_ar_seg", 3 * 640 * 480 + 16);
    if (!reader.open())
    {
        std::cerr << "warning: seg shm open failed" << std::endl;
        return;
    }

    while (!g_exit)
    {
        cv::Mat seg_img;
        if (!reader.read_mask(seg_img))
        {
            usleep(kSegShmIdleUs);
            continue;
        }

        int target_w = 0;
        int target_h = 0;
        {
            std::lock_guard<std::mutex> lock(mtx_frame);
            if (!latest_frame.empty())
            {
                target_w = latest_frame.cols;
                target_h = latest_frame.rows;
            }
        }

        if (target_w > 0 && target_h > 0 && (seg_img.cols != target_w || seg_img.rows != target_h))
        {
            cv::Mat resized;
            cv::resize(seg_img, resized, cv::Size(target_w, target_h), 0, 0, cv::INTER_NEAREST);
            seg_img = resized;
        }

        // SHM 源图像已是正立，Python 推理输出的 mask 无需翻转
        {
            lock_guard<mutex> lock(mtx_produce);
            G_produce.seg_mask = seg_img;
            g_seg_seq.fetch_add(1, std::memory_order_release);
        }
    }

    reader.close();
}

void stream_thread(Produce &G_produce)
{
    PIN_TO_CORE(2); // MJPEG 推流非实时，放小核 core 2
    // 显示线程：仅负责把最新帧和检测结果渲染到 MJPEG 流，不参与控制逻辑。
    MjpegStreamer streamer(MJPEG_PORT, MJPEG_QUALITY, STREAM_IDLE_US);
    if (!streamer.start())
    {
        std::cerr << "failed to start mjpeg server on port " << MJPEG_PORT << std::endl;
        g_exit = true;
        return;
    }

    std::cout << "mjpeg stream ready: http://<device-ip>:" << MJPEG_PORT << "/" << std::endl;
    std::cout << "press q/h/e/ESC then Enter to exit" << std::endl;

    double stream_fps = 0.0;
    uint32_t stream_frame_count = 0;
    auto stream_last_tick = std::chrono::steady_clock::now();
    uint64_t last_stream_seq = 0;
    // 预分配渲染缓冲，避免每帧 900KB 克隆
    cv::Mat stream_buffer;

    while (!g_exit)
    {
        MjpegStreamer::pollExitKey(g_exit, stop_flag);

        // 从 G_produce 读取（暂回退，待排查段错误后重新启用独立缓冲）
        const uint64_t current_seq = g_produce_seq.load(std::memory_order_acquire);
        if (current_seq == last_stream_seq)
        {
            usleep(STREAM_IDLE_US);
            continue;
        }
        last_stream_seq = current_seq;

        Produce produce_snapshot;
        {
            lock_guard<mutex> lock(mtx_produce);
            produce_snapshot = G_produce;
        }

        if (produce_snapshot.img.empty())
        {
            usleep(STREAM_IDLE_US);
            continue;
        }

        // 复用预分配缓冲（仅首帧或尺寸变化时分配）
        if (stream_buffer.empty() || stream_buffer.size() != produce_snapshot.img.size())
            stream_buffer.create(produce_snapshot.img.size(), produce_snapshot.img.type());
        produce_snapshot.img.copyTo(stream_buffer);
        cv::Mat stream_img = stream_buffer;

        if (!produce_snapshot.morph_mask.empty())
        {
            cv::Mat mask_bgr;
            if (produce_snapshot.morph_mask.channels() == 1)
            {
                cv::cvtColor(produce_snapshot.morph_mask, mask_bgr, cv::COLOR_GRAY2BGR);
            }
            else
            {
                mask_bgr = produce_snapshot.morph_mask;
            }

            if (mask_bgr.size() == stream_img.size())
            {
                cv::Mat green_mask = cv::Mat::zeros(mask_bgr.size(), CV_8UC3);
                std::vector<cv::Mat> channels(3);
                cv::split(mask_bgr, channels);
                green_mask.setTo(cv::Scalar(0, 255, 0), channels[0] > 0);
                cv::addWeighted(stream_img, 0.70, green_mask, 0.30, 0.0, stream_img);
            }
        }

        drawBox(stream_img, produce_snapshot.predict_results);

        char fps_text[128];
        std::snprintf(fps_text,
                      sizeof(fps_text),
                      "stream fps: %.1f | run fps: %.1f | error:%.1f",
                      stream_fps,
                      g_run_thread_fps.load(std::memory_order_relaxed), g_error.load(std::memory_order_relaxed));
        cv::putText(stream_img,
                    fps_text,
                    cv::Point(16, 30),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.8,
                    cv::Scalar(0, 255, 0),
                    2,
                    cv::LINE_AA);

        if (!streamer.processFrame(stream_img))
        {
            usleep(STREAM_IDLE_US);
            continue;
        }

        ++stream_frame_count;
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(now - stream_last_tick).count();
        if (elapsed >= 1.0)
        {
            stream_fps = stream_frame_count / elapsed;
            stream_frame_count = 0;
            stream_last_tick = now;
        }
    }

    streamer.stop();
}

void drawBox(Mat &img, const std::vector<PredictResult> &predict_results)

{
    // 在调试流中把检测框和类别名绘制到图像上。

    const size_t n = predict_results.size();

    std::vector<cv::Point> temp_point;
    std::vector<cv::Point> t_temp_point;

    for (size_t i = 0; i < n; ++i)
    {

        const PredictResult &result = predict_results[i];

        int cx = (result.x1 + result.x2) / 2;
        int cy = result.y2 + 10;
        temp_point.push_back(cv::Point(cx, cy));
        cv::Point2f t_pt = transf(cx, cy);
        t_temp_point.push_back(cv::Point(t_pt.x, t_pt.y));

        auto score = std::to_string(result.score);

        int pointY = result.y1 - 20;

        if (pointY < 0)

            pointY = 0;

        const int box_w = std::max(1, result.x2 - result.x1);

        const int box_h = std::max(1, result.y2 - result.y1);

        cv::Rect rectText(result.x1, pointY, box_w, 20);

        cv::rectangle(img, rectText, getBoxColor(result.class_id), -1);

        const std::size_t dot_pos = score.find('.');

        const std::string score_text = (dot_pos == std::string::npos || dot_pos + 3 >= score.size())

                                           ? score

                                           : score.substr(0, dot_pos + 3);

        std::string label_name = "cls=" + std::to_string(result.class_id) + " [" + score_text + "]";

        cv::Rect rect(result.x1, result.y1, box_w, box_h);

        cv::rectangle(img, rect, getBoxColor(result.class_id), 1);

        cv::putText(img, label_name, Point(result.x1, result.y1), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 254), 1);
    }

    // 逆透视目标检测画点
    for (size_t i = 0; i < t_temp_point.size(); ++i)
    {
        cv::circle(img, t_temp_point[i], 4, cv::Scalar(0, 0, 255), -1);
    }

    std::vector<Point> local_draw_line;
    {
        extern std::mutex mtx_draw;
        std::lock_guard<std::mutex> lock(mtx_draw);
        local_draw_line = draw_line;
    }

    // filtered_line_CV绘图
    for (size_t i = 0; i + 1 < local_draw_line.size(); ++i)
    {
        cv::line(img, local_draw_line[i], local_draw_line[i + 1], cv::Scalar(255, 0, 0), 3);
    }
    for (size_t i = 0; i + 1 < local_draw_line.size(); ++i)
    {
        cv::circle(img, local_draw_line[i], 4, cv::Scalar(0, 255, 255), -1);
    }
    cv::Point aim_pt = g_aim_point;
    if (aim_pt.x > 0 && aim_pt.y > 0)
    {
        cv::circle(img, aim_pt, 8, cv::Scalar(255, 255, 255), -1); // 红色实心
        // cv::putText(img, "AIM", aim_pt + cv::Point(12, -8),
        //             cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
    }

    // // 绘制预瞄点
    // {
    //     extern std::mutex mtx_draw;
    //     std::lock_guard<std::mutex> lock(mtx_draw);
    //     cv::Point aim_pt = g_aim_point;
    //     if (aim_pt.x > 0 && aim_pt.y > 0)
    //     {
    //         cv::circle(img, aim_pt, 8, cv::Scalar(255, 255, 255), -1); // 红色实心
    //         // cv::putText(img, "AIM", aim_pt + cv::Point(12, -8),
    //         //             cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
    //     }
    // }
}
