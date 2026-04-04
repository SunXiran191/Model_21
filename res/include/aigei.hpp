#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "json.hpp"

using json = nlohmann::json;

/// @brief 
/// @param expected_ip result来源ip 
/// @param expected_port 端口
/// @return /
inline std::vector<float> result_get(const std::string& expected_ip, int expected_port) {
    int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) {
        std::cerr << "Error: Cannot create socket\n";                   
        return {};
    }
             
    int reuse = 1;
    setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(static_cast<unsigned short>(expected_port));

    if (bind(sockfd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        std::cerr << "Error: bind failed\n";
        close(sockfd);
        return {};
    }

    char buffer[2048] = {0};
    sockaddr_in src{};
    socklen_t src_len = sizeof(src);
    const int n = recvfrom(sockfd, buffer, sizeof(buffer) - 1, 0,
                           reinterpret_cast<sockaddr*>(&src), &src_len);
    if (n <= 0) {
        std::cerr << "Error: recvfrom failed\n";
        close(sockfd);
        return {};
    }

    const char* src_ip = inet_ntoa(src.sin_addr);
    const int src_port = ntohs(src.sin_port);
    if (std::strcmp(src_ip, expected_ip.c_str()) != 0 || src_port != expected_port) {
        std::cerr << "Error: packet source mismatch, from "
                  << src_ip << ":" << src_port << "\n";
        close(sockfd);
        return {};
    }

    buffer[n] = '\0';

    std::vector<float> result;
    try {
        const json j = json::parse(buffer);        //错误会抛出异常
        if (j.contains("result") && j["result"].is_array()) {
            result = j["result"].get<std::vector<float>>();
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: JSON parse failed: " << e.what() << "\n";
        close(sockfd);
        return {};
    }

    close(sockfd);
    return result;

}

