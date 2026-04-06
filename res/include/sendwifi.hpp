#pragma once

#include <string>
#include <cstring>
#include <iostream>
#include <cstdint>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <unistd.h>

/**
 * @brief sendwifi 类 — 通过 UDP 将速度和角度以二进制帧发送到指定 IP:port
 * 帧格式与现有串口协议一致：
 * [0]    : 0x42 (帧头)
 * [1]    : 0x01 (地址)
 * [2]    : 12   (帧长)
 * [3..6] : float speed (4 bytes, 小端)
 * [7..10]: float angle (4 bytes, 小端)
 * [11]   : checksum (前 11 字节累加)
 */
class sendWIFI {
public:
  sendWIFI(const std::string &ip, uint16_t port) : ip_(ip), port_(port), sockfd_(-1), opened_(false) {}
  ~sendWIFI() { close(); }

  // 打开 UDP socket
  bool open()
  {
    if (opened_)
      return true;

    sockfd_ = ::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (sockfd_ < 0)
    {
      std::cerr << "socket() failed" << std::endl;
      return false;
    }

    std::memset(&destAddr_, 0, sizeof(destAddr_));
    destAddr_.sin_family = AF_INET;
    destAddr_.sin_port = htons(port_);
    int r = inet_pton(AF_INET, ip_.c_str(), &destAddr_.sin_addr);
    if (r != 1)
    {
      std::cerr << "inet_pton failed for ip: " << ip_ << std::endl;
      ::close(sockfd_);
      sockfd_ = -1;
      return false;
    }

    opened_ = true;
    return true;
  }

  // 关闭 socket
  void close()
  {
    if (!opened_)
      return;
    ::close(sockfd_);
    sockfd_ = -1;
    opened_ = false;
  }

  // 发送速度与角度（返回 0 表示成功）
  int send(float speed, float angle)
  {
    if (!opened_ && !open())
      return -1;

    uint8_t buff[12];
    uint8_t check = 0;
    Bit32Union b1, b2;

    buff[0] = 0x42;
    buff[1] = 0x01;
    buff[2] = 12;

    b1.float32 = speed;
    for (int i = 0; i < 4; ++i)
      buff[3 + i] = b1.buff[i];

    b2.float32 = angle;
    for (int i = 0; i < 4; ++i)
      buff[7 + i] = b2.buff[i];

    for (int i = 0; i < 11; ++i)
      check += buff[i];
    buff[11] = check;

    ssize_t sent = ::sendto(sockfd_, reinterpret_cast<const void *>(buff), sizeof(buff), 0,
                             reinterpret_cast<struct sockaddr *>(&destAddr_), sizeof(destAddr_));
    if (sent == (ssize_t)sizeof(buff))
      return 0;

    std::cerr << "sendto failed or sent partial: " << sent << std::endl;
    return -2;
  }

private:
  std::string ip_;
  uint16_t port_;
  int sockfd_;
  struct sockaddr_in destAddr_;
  bool opened_ = false;

  typedef union
  {
    uint8_t buff[4];
    float float32;
    int32_t int32;
  } Bit32Union;
};
