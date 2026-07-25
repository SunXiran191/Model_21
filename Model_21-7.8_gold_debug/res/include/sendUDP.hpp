#pragma once

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cstring>

class UdpTelemetrySender {
public:
    UdpTelemetrySender(const std::string& ip, int port)
        : is_initialized_(false), socket_open_(false), sock_(0), server_addr_{} {
        if (!create_socket_()) {
            std::cerr << "[UDP Sender] init failed: could not create socket\n";
            return;
        }
        if (!set_server_addr_(ip, port)) {
            std::cerr << "[UDP Sender] init failed: invalid address " << ip << ":" << port << "\n";
            close_socket_();
            return;
        }
        is_initialized_ = true;
        std::cout << "[UDP Sender] ready, sending to " << ip << ":" << port << std::endl;
    }

    ~UdpTelemetrySender() {
        close_socket_();
    }

    void send_data(double error, double xerror) {
        if (!is_initialized_) {
            return;
        }

        const std::string msg = build_json_(error, xerror);
        const int sent = sendto(sock_, msg.c_str(), msg.size(), 0,
                                reinterpret_cast<const sockaddr*>(&server_addr_),
                                sizeof(server_addr_));
        if (sent < 0) {
            std::cerr << "[UDP Sender] send failed: " << std::strerror(errno) << " (errno=" << errno << ")\n";
        }
    }

private:
    static double now_seconds_() {
        const auto now = std::chrono::system_clock::now().time_since_epoch();
        return std::chrono::duration_cast<std::chrono::milliseconds>(now).count() / 1000.0;
    }

    static double round3_(double v) {
        return std::round(v * 1000.0) / 1000.0;
    }

    static std::string build_json_(double error,
                                   double xerror) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(3)
            << "{\"timestamp\":" << round3_(now_seconds_()) << ","
            << "\"error\":" << round3_(error) << ","
            << "\"xerror\":" << round3_(xerror) << "}";
        return oss.str();
    }

    bool create_socket_() {
        sock_ = ::socket(AF_INET, SOCK_DGRAM, 0);
        if (sock_ < 0) {
            std::cerr << "[UDP Sender] Socket create failed\n";
            return false;
        }
        socket_open_ = true;
        return true;
    }

    bool set_server_addr_(const std::string& ip, int port) {
        server_addr_.sin_family = AF_INET;
        server_addr_.sin_port = htons(static_cast<unsigned short>(port));
        if (inet_aton(ip.c_str(), &server_addr_.sin_addr) == 0) {
            std::cerr << "[UDP Sender] Invalid IP: " << ip << "\n";
            return false;
        }
        return true;
    }

    void close_socket_() {
        if (!socket_open_) {
            return;
        }
        close(sock_);
        socket_open_ = false;
        is_initialized_ = false;
    }

    bool is_initialized_;
    bool socket_open_;
    int sock_;
    sockaddr_in server_addr_;
};
