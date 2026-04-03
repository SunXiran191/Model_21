#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>
#include <thread>

#define USB_FRAME_HEAD 0x42
#define USB_FRAME_LENMIN 8
#define USB_FRAME_LENMAX 12

#define USB_ADDR_CARCTRL 1
#define USB_ADDR_BUZZER 4
#define USB_ADDR_LED 5
#define USB_ADDR_KEY 6

enum Buzzer {
    BUZZER_OK = 0,
    BUZZER_WARNNING ,
    BUZZER_FINISH ,
    BUZZER_DING ,
    BUZZER_START ,
};

class Uart {
public:
    explicit Uart(const std::string& port)
        : keypress(false), pitch_angle(0.0), port_name_(port), is_open_(false), running_(false) {}

    ~Uart() {
        close();
    }

    int open() {
        // Placeholder: serial backend is platform-specific.
        is_open_ = true;
        std::cout << "Uart open placeholder on port " << port_name_ << "\n";
        return 0;
    }

    int receive_bytes(int timeout_ms = 0) {
        (void)timeout_ms;
        if (!is_open_) {
            return -1;
        }
        return -1;
    }

    int transmit_byte(int data) {
        if (!is_open_) {
            return 0;
        }
        (void)data;
        return 1;
    }

    void start_receive() {
        if (!is_open_) {
            return;
        }

        running_.store(true);
        thread_rec_ = std::thread(&Uart::_receive_loop, this);
    }

    void close() {
        running_.store(false);
        if (thread_rec_.joinable()) {
            thread_rec_.join();
        }
        is_open_ = false;
    }

    bool keypress;
    double pitch_angle;

private:
    void _receive_loop() {
        while (running_.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    std::string port_name_;
    bool is_open_;
    std::thread thread_rec_;
    std::atomic<bool> running_;
};
