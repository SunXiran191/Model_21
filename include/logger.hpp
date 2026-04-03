#pragma once

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>

class Logger {
public:
    explicit Logger(const std::string& filename, int log_interval_ms = 0)
        : interval_ms_(log_interval_ms), last_log_time_ms_(now_ms()) {
        log_file_.open(filename, std::ios::app);
        if (!log_file_.is_open()) {
            std::cerr << "Error: Cannot open log file: " << filename << "\n";
            return;
        }
        log_file_ << "AbsoluteTime(ms),X,Y,Angle,Speed\n";
    }

    ~Logger() {
        if (log_file_.is_open()) {
            log_file_.close();
        }
    }

    void log(double x, double y, double angle, double speed) {
        if (!log_file_.is_open()) {
            return;
        }

        const std::int64_t now = now_ms();
        if (interval_ms_ > 0 && (now - last_log_time_ms_) < interval_ms_) {
            return;
        }
        last_log_time_ms_ = now;

        std::lock_guard<std::mutex> lock(log_mutex_);
        log_file_ << now << ",";
        log_file_.setf(std::ios::fixed);
        log_file_.precision(2);
        log_file_ << x << ",";
        log_file_.precision(6);
        log_file_ << y << "," << angle << "," << speed << "\n";
    }

private:
    static std::int64_t now_ms() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::system_clock::now().time_since_epoch())
            .count();
    }

    int interval_ms_;
    std::mutex log_mutex_;
    std::ofstream log_file_;
    std::int64_t last_log_time_ms_;
};
