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

class UdpTelemetrySender {
public:
    UdpTelemetrySender(const std::string& ip, int port)
        : is_initialized_(false), socket_open_(false), sock_(0), server_addr_{} {
        if (!create_socket_()) {
            return;
        }
        if (!set_server_addr_(ip, port)) {
            close_socket_();
            return;
        }
        is_initialized_ = true;
    }

    ~UdpTelemetrySender() {
        close_socket_();
    }

    void send_data(const std::string& id_str, double x, double y, double z, double yaw) {
        if (!is_initialized_) {
            return;
        }

        const std::string msg = build_json_(id_str, x, y, z, yaw);
        const int sent = sendto(sock_, msg.c_str(), msg.size(), 0,
                                reinterpret_cast<const sockaddr*>(&server_addr_),
                                sizeof(server_addr_));
        if (sent < 0) {
            std::cerr << "[UDP Sender] send failed\n";
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

    static std::string build_json_(const std::string& id_str,
                                   double x,
                                   double y,
                                   double z,
                                   double yaw) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(3)
            << "{\"id\":\"" << id_str << "\"," 
            << "\"timestamp\":" << round3_(now_seconds_()) << ","
            << "\"x\":" << round3_(x) << ","
            << "\"y\":" << round3_(y) << ","
            << "\"z\":" << round3_(z) << ","
            << "\"pitch\":0.0,"
            << "\"yaw\":" << round3_(yaw) << ","
            << "\"roll\":0.0}";
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
