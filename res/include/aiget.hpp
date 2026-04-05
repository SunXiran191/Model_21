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

class aiget{
public:
struct PredictResult {
    int x1; //左上角
    int y1;
    int x2; //右下角
    int y2;
    float score; // 置信度
    int class_id;
};

// 与 UDP JSON 对齐：一帧里 objects 数组可包含多个检测目标。
struct PredictFrame {
    int count = 0;
    std::vector<PredictResult> objects;
};

/// @brief 返回包含整帧检测结果的结构体（支持同帧多个目标）
/// @param expected_ip result来源ip
/// @param expected_port 端口
/// @return PredictFrame
inline PredictFrame result_get_frame(const std::string& expected_ip, int expected_port) {
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
    if (std::strcmp(src_ip, expected_ip.c_str()) != 0) {
        std::cerr << "Error: packet source mismatch, from "
                  << src_ip << ":" << src_port << "\n";
        close(sockfd);
        return {};
    }

    buffer[n] = '\0';

    PredictFrame frame;
    try {
        const json j = json::parse(buffer);
        if (j.contains("count") && j["count"].is_number_integer()) {
            frame.count = j["count"].get<int>();
        }

        if (j.contains("objects") && j["objects"].is_array()) {
            for (const auto& obj : j["objects"]) {
                if (!obj.is_object()) {
                    continue;
                }

                if (!obj.contains("x1") || !obj.contains("y1") ||
                    !obj.contains("x2") || !obj.contains("y2") ||
                    !obj.contains("score") || !obj.contains("class_id")) {
                    continue;
                }

                PredictResult det{};
                det.x1 = obj["x1"].get<int>();
                det.y1 = obj["y1"].get<int>();
                det.x2 = obj["x2"].get<int>();
                det.y2 = obj["y2"].get<int>();
                det.score = obj["score"].get<float>();
                det.class_id = obj["class_id"].get<int>();
                frame.objects.push_back(det);
            }
        }

        // 若发送端未提供 count，则使用 objects 实际长度回填。
        if (frame.count <= 0) {
            frame.count = static_cast<int>(frame.objects.size());
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: JSON parse failed: " << e.what() << "\n";
        close(sockfd);
        return {};
    }

    close(sockfd);
    return frame;
}

/// @brief 返回PredictResult结构体
/// @param expected_ip result来源ip 
/// @param expected_port 端口
/// @return /
inline std::vector<PredictResult> result_get_struct(const std::string& expected_ip, int expected_port) {
    return result_get_frame(expected_ip, expected_port).objects;
}



/// @brief 
/// @param expected_ip result来源ip 
/// @param expected_port 端口
/// @return /
inline std::vector<float> result_get(const std::string& expected_ip, int expected_port) {
    std::vector<float> result;
    const PredictFrame frame = result_get_frame(expected_ip, expected_port);

    // count 放在最前面，再依次拼接每个目标的 6 个字段。
    result.push_back(static_cast<float>(frame.count));
    for (const auto& det : frame.objects) {
        result.push_back(static_cast<float>(det.x1));
        result.push_back(static_cast<float>(det.y1));
        result.push_back(static_cast<float>(det.x2));
        result.push_back(static_cast<float>(det.y2));
        result.push_back(det.score);
        result.push_back(static_cast<float>(det.class_id));
    }
    return result;
}

};