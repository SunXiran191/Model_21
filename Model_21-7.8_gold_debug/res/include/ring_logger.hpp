#pragma once

/**
 * ring_logger.hpp — 环形缓冲区日志器
 * ====================================
 * 作用：将高频日志（每帧 printf）先存入环形缓冲区，
 *       定时（每秒）批量 flush 输出，大幅减少系统调用开销。
 *
 * 用法：
 *   RingLogger& log = RingLogger::instance();
 *   log.info("speed: %.2f", speed);   // 存入缓冲区
 *   log.flush();                       // 批量输出（可放在每秒一次的定时器中）
 *
 * 线程安全：内部使用 mutex 保护。
 */

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

class RingLogger
{
public:
    static RingLogger &instance()
    {
        static RingLogger inst;
        return inst;
    }

    // printf 风格写入环形缓冲区
    void info(const char *fmt, ...)
    {
        va_list args;
        va_start(args, fmt);
        char buf[256];
        vsnprintf(buf, sizeof(buf), fmt, args);
        va_end(args);

        std::lock_guard<std::mutex> lock(mtx_);
        if (count_ < capacity_)
        {
            buffer_[count_++] = buf;
        }
        else
        {
            // 缓冲区已满，覆盖最旧的
            buffer_[next_ % capacity_] = buf;
            ++next_;
            if (count_ < capacity_)
                ++count_;
        }
    }

    // 将缓冲区所有内容一次性输出到 stdout 并清空
    void flush()
    {
        std::lock_guard<std::mutex> lock(mtx_);
        if (count_ == 0)
            return;

        // 计算实际要输出的行数上限，防止单次 flush 输出过多
        const size_t kMaxFlushLines = 120;
        size_t to_print = (count_ > kMaxFlushLines) ? kMaxFlushLines : count_;

        if (count_ > kMaxFlushLines)
        {
            printf("[RingLogger] *** %zu lines suppressed ***\n", count_ - kMaxFlushLines);
        }

        if (count_ <= capacity_)
        {
            // 正常顺序：buffer_[0..count_-1]
            for (size_t i = 0; i < to_print; ++i)
            {
                printf("%s", buffer_[i].c_str());
            }
        }
        else
        {
            // 环形覆盖后，从 next_ 开始输出到末尾，再从头到 next_%capacity_
            size_t start = next_ % capacity_;
            for (size_t i = 0; i < to_print; ++i)
            {
                size_t idx = (start + i) % capacity_;
                printf("%s", buffer_[idx].c_str());
            }
        }
        fflush(stdout);
        count_ = 0;
        next_ = 0;
    }

    // 获取当前缓冲行数
    size_t size() const
    {
        std::lock_guard<std::mutex> lock(mtx_);
        return count_;
    }

private:
    RingLogger() : capacity_(512), count_(0), next_(0)
    {
        buffer_.resize(capacity_);
    }
    ~RingLogger() = default;
    RingLogger(const RingLogger &) = delete;
    RingLogger &operator=(const RingLogger &) = delete;

    size_t capacity_;
    size_t count_; // 实际存储的行数
    size_t next_;  // 环形写入位置（仅 count_>capacity_ 时有意义）
    std::vector<std::string> buffer_;
    mutable std::mutex mtx_;
};

// 便捷宏：全局开关，可在编译时关闭所有日志
#ifndef LOG_DISABLE
#define LOG_INFO(...) RingLogger::instance().info(__VA_ARGS__)
#define LOG_FLUSH() RingLogger::instance().flush()
#else
#define LOG_INFO(...) ((void)0)
#define LOG_FLUSH() ((void)0)
#endif
