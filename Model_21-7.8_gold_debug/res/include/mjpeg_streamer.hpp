#pragma once

#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstring>
#include <string>
#include <vector>

#include <unistd.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <fcntl.h>

#include <opencv2/opencv.hpp>

class MjpegStreamer
{
public:
    MjpegStreamer(int port, int quality, useconds_t idle_us)
        : port_(port), quality_(quality), idle_us_(idle_us), server_fd_(-1)
    {
    }

    ~MjpegStreamer()
    {
        stop();
    }

    bool start()
    {
        // 忽略 SIGPIPE，防止往已关闭的 socket 写入时进程被杀死
        signal(SIGPIPE, SIG_IGN);
        server_fd_ = createServer(port_);
        return server_fd_ >= 0;
    }

    void stop()
    {
        for (size_t i = 0; i < client_fds_.size(); ++i)
        {
            if (client_fds_[i] >= 0)
                close(client_fds_[i]);
        }
        client_fds_.clear();
        if (server_fd_ >= 0)
        {
            close(server_fd_);
            server_fd_ = -1;
        }
    }

    int port() const
    {
        return port_;
    }

    useconds_t idleUs() const
    {
        return idle_us_;
    }

    bool processFrame(const cv::Mat &img)
    {
        if (server_fd_ < 0)
        {
            return false;
        }

        // 始终尝试接受新客户端
        acceptClient();

        if (img.empty())
        {
            usleep(idle_us_);
            return true;
        }

        if (client_fds_.empty())
        {
            usleep(idle_us_);
            return true;
        }

        // 编码一次 JPEG，所有客户端共享
        std::vector<uchar> jpg;
        std::vector<int> encode_params;
        encode_params.push_back(cv::IMWRITE_JPEG_QUALITY);
        encode_params.push_back(quality_);

        if (!cv::imencode(".jpg", img, jpg, encode_params))
        {
            usleep(idle_us_);
            return true;
        }

        const std::string part_header =
            "--frame\r\n"
            "Content-Type: image/jpeg\r\n"
            "Content-Length: " +
            std::to_string(jpg.size()) + "\r\n\r\n";

        // 非阻塞发送给所有客户端，慢客户端跳过该帧但保持连接
        std::vector<int> alive;
        alive.reserve(client_fds_.size());
        for (size_t i = 0; i < client_fds_.size(); ++i)
        {
            int fd = client_fds_[i];
            if (fd < 0)
                continue;
            bool ok = sendAll(fd, part_header.data(), part_header.size()) &&
                      sendAll(fd, reinterpret_cast<const char *>(jpg.data()), jpg.size()) &&
                      sendAll(fd, "\r\n", 2);
            if (!ok)
            {
                if (errno == EAGAIN || errno == EWOULDBLOCK)
                {
                    // 客户端 TCP 缓冲区满，保留连接下帧再试
                    alive.push_back(fd);
                    continue;
                }
                // 连接已断开（EPIPE/ECONNRESET 等）
                close(fd);
            }
            else
            {
                alive.push_back(fd);
            }
        }
        client_fds_.swap(alive);

        return true;
    }

    static bool pollExitKey(std::atomic<bool> &exit_flag, std::atomic<bool> &stop_flag)
    {
        fd_set readfds;
        FD_ZERO(&readfds);
        FD_SET(STDIN_FILENO, &readfds);

        struct timeval tv;
        tv.tv_sec = 0;
        tv.tv_usec = 0;

        const int ret = select(STDIN_FILENO + 1, &readfds, NULL, NULL, &tv);
        if (ret <= 0 || !FD_ISSET(STDIN_FILENO, &readfds))
        {
            return false;
        }

        char ch = 0;
        const ssize_t n = read(STDIN_FILENO, &ch, 1);
        if (n <= 0)
        {
            return false;
        }
        if(ch == 'w')
        {
            stop_flag = false;
        }
        if(ch == 's')
        {
            stop_flag = 1;
        }
        if (ch == 27 || ch == 'q' || ch == 'Q')
        {
            exit_flag = true;
            return true;
        }

        return false;
    }

private:
    // 非阻塞发送：EAGAIN 时保留 errno 让调用方判断
    static bool sendAll(int fd, const char *data, size_t len)
    {
        while (len > 0)
        {
            ssize_t sent = send(fd, data, len, MSG_NOSIGNAL);
            if (sent < 0)
            {
                if (errno == EINTR)
                    continue;
                // EAGAIN/EWOULDBLOCK: 非阻塞模式缓冲区满，保留 errno 返回 false
                return false;
            }
            if (sent == 0)
            {
                return false;
            }
            data += sent;
            len -= static_cast<size_t>(sent);
        }
        return true;
    }

    static int createServer(int port)
    {
        int fd = socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0)
        {
            return -1;
        }

        int reuse = 1;
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

        sockaddr_in addr;
        std::memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_port = htons(static_cast<uint16_t>(port));
        addr.sin_addr.s_addr = htonl(INADDR_ANY);

        if (bind(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0)
        {
            close(fd);
            return -1;
        }

        if (listen(fd, 8) < 0)
        {
            close(fd);
            return -1;
        }

        int flags = fcntl(fd, F_GETFL, 0);
        if (flags >= 0)
        {
            fcntl(fd, F_SETFL, flags | O_NONBLOCK);
        }

        return fd;
    }

    void acceptClient()
    {
        sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int fd = accept(server_fd_, reinterpret_cast<sockaddr *>(&client_addr), &client_len);
        if (fd < 0)
        {
            return;
        }

        // 设客户端 socket 为非阻塞，防止慢客户端拖死整个推流线程
        int fl = fcntl(fd, F_GETFL, 0);
        if (fl >= 0)
            fcntl(fd, F_SETFL, fl | O_NONBLOCK);

        // 禁用 Nagle 算法，确保 MJPEG 帧立即发送不延迟
        int tcp_nodelay = 1;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &tcp_nodelay, sizeof(tcp_nodelay));

        // 启用 TCP keepalive，及时检测断开的客户端
        int keepalive = 1;
        setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &keepalive, sizeof(keepalive));

        const std::string http_header =
            "HTTP/1.1 200 OK\r\n"
            "Server: line-tracker\r\n"
            "Cache-Control: no-cache, no-store, must-revalidate\r\n"
            "Pragma: no-cache\r\n"
            "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n";

        if (!sendAll(fd, http_header.data(), http_header.size()))
        {
            close(fd);
            return;
        }

        // 限制最大客户端数，防止资源耗尽
        if (client_fds_.size() >= 8)
        {
            close(client_fds_.front());
            client_fds_.erase(client_fds_.begin());
        }

        client_fds_.push_back(fd);
    }

    void closeClient(int fd)
    {
        if (fd >= 0)
        {
            close(fd);
        }
    }

private:
    int port_;
    int quality_;
    useconds_t idle_us_;
    int server_fd_;
    std::vector<int> client_fds_;
};
