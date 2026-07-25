#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>

#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

#define USB_FRAME_HEAD 0x42
#define USB_FRAME_LENMIN 8
#define USB_FRAME_LENMAX 12

#define USB_ADDR_CARCTRL 1
#define USB_ADDR_BUZZER 4
#define USB_ADDR_LED 5
#define USB_ADDR_KEY 6

enum Buzzer
{
    BUZZER_OK = 0,
    BUZZER_WARNNING,
    BUZZER_FINISH,
    BUZZER_DING,
    BUZZER_START,
};

class Uart
{
public:
    explicit Uart(const std::string &port)
        : keypress(false), pitch_angle(0.0), port_name_(port),
          is_open_(false), running_(false), fd_(-1), dump_mode_(false) {}

    ~Uart()
    {
        close();
    }

    int open()
    {
        // Placeholder: serial backend is platform-specific.
        is_open_ = true;
        std::cout << "Uart open placeholder on port " << port_name_ << "\n";
        return 0;
    }

    int receive_bytes(int timeout_ms = 0)
    {
        (void)timeout_ms;
        if (!is_open_)
        {
            return -1;
        }
        return -1;
    }

    int transmit_byte(int data)
    {
        if (!is_open_)
        {
            return 0;
        }
        (void)data;
        return 1;
    }

    void start_receive()
    {
        if (!is_open_)
        {
            return;
        }

        running_.store(true);
        thread_rec_ = std::thread(&Uart::_receive_loop, this);
    }

    void close()
    {
        running_.store(false);
        if (thread_rec_.joinable())
        {
            thread_rec_.join();
        }
        if (fd_ >= 0)
        {
            ::close(fd_);
            fd_ = -1;
        }
        is_open_ = false;
    }

    bool keypress;
    double pitch_angle;

    // Receive speed frame from lower controller.
    // Returns true when a valid frame is parsed and speed is updated.
    bool receive_speed(float &speed, int timeout_ms = 100)
    {
        if (!is_open_ || fd_ < 0)
        {
            return false;
        }

        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::milliseconds(timeout_ms > 0 ? timeout_ms : 0);

        while (true)
        {
            uint8_t temp[128];
            const int n = (int)::read(fd_, temp, sizeof(temp));
            if (n > 0)
            {
                const size_t copy_n = (rx_len_ + (size_t)n > sizeof(rx_buf_))
                                          ? (sizeof(rx_buf_) - rx_len_)
                                          : (size_t)n;
                if (copy_n > 0)
                {
                    std::memcpy(rx_buf_ + rx_len_, temp, copy_n);
                    rx_len_ += copy_n;
                }

                if (_parse_speed_frame(speed))
                {
                    return true;
                }
            }
            else
            {
                if (timeout_ms <= 0 || std::chrono::steady_clock::now() >= deadline)
                {
                    return false;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    }

    // ============================================================
    //  ���ݼ��� / ת��ģʽ �� ���ڷ�����λ�����ݸ�ʽ
    // ============================================================

    /// @brief ��ԭʼ����ת��ģʽ�򿪴��ڣ�ʹ�� POSIX termios��
    /// @param baud �����ʣ��� B115200, B9600 ��
    /// @return 0=�ɹ�, -1=ʧ��
    int open_dump(speed_t baud = B115200)
    {
        fd_ = ::open(port_name_.c_str(), O_RDWR | O_NOCTTY | O_NDELAY);
        if (fd_ < 0)
        {
            std::cerr << "[Uart::open_dump] Cannot open " << port_name_ << std::endl;
            return -1;
        }

        struct termios tty;
        std::memset(&tty, 0, sizeof(tty));
        if (tcgetattr(fd_, &tty) != 0)
        {
            std::cerr << "[Uart::open_dump] tcgetattr failed" << std::endl;
            ::close(fd_);
            fd_ = -1;
            return -1;
        }

        cfsetospeed(&tty, baud);
        cfsetispeed(&tty, baud);

        tty.c_cflag |= (CLOCAL | CREAD); // ���ý���
        tty.c_cflag &= ~CSIZE;
        tty.c_cflag |= CS8;      // 8 λ����
        tty.c_cflag &= ~PARENB;  // ��У��
        tty.c_cflag &= ~CSTOPB;  // 1 λֹͣλ
        tty.c_cflag &= ~CRTSCTS; // ��Ӳ������

        tty.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG); // ԭʼģʽ
        tty.c_iflag &= ~(IXON | IXOFF | IXANY);         // ����������
        tty.c_iflag &= ~(INLCR | ICRNL | IGNCR);        // ��ת������
        tty.c_oflag &= ~OPOST;                          // ԭʼ���

        tty.c_cc[VMIN] = 0;  // ��������
        tty.c_cc[VTIME] = 1; // 100ms ��ʱ

        tcflush(fd_, TCIOFLUSH);
        if (tcsetattr(fd_, TCSANOW, &tty) != 0)
        {
            std::cerr << "[Uart::open_dump] tcsetattr failed" << std::endl;
            ::close(fd_);
            fd_ = -1;
            return -1;
        }

        is_open_ = true;
        std::cout << "[Uart::open_dump] Serial port " << port_name_
                  << " opened (fd=" << fd_ << ")" << std::endl;
        return 0;
    }

    /// @brief ��������ת���߳� �� ������ȡ����ӡ����ԭʼ�ֽ�
    /// @note �ȵ��� open_dump()���ٵ�������
    void start_dump()
    {
        if (!is_open_ || fd_ < 0)
        {
            std::cerr << "[Uart::start_dump] Port not opened yet!" << std::endl;
            return;
        }
        dump_mode_.store(true);
        running_.store(true);
        thread_rec_ = std::thread(&Uart::_dump_loop, this);
        std::cout << "[Uart::start_dump] Dump thread started, waiting for data..."
                  << std::endl;
    }

    /// @brief ֹͣת��
    void stop_dump()
    {
        dump_mode_.store(false);
        running_.store(false);
    }

    /// @brief �Ƿ�����ת��
    bool is_dumping() const { return dump_mode_.load(); }

private:
    // ============================================================
    //  ԭʼ����ת��ѭ��
    // ============================================================
    void _dump_loop()
    {
        const int BUF_SIZE = 1024;
        uint8_t buf[BUF_SIZE];
        uint64_t total_bytes = 0;
        auto start_time = std::chrono::steady_clock::now();

        while (running_.load() && dump_mode_.load())
        {
            int n = (int)::read(fd_, buf, BUF_SIZE);
            if (n > 0)
            {
                total_bytes += n;
                auto now = std::chrono::steady_clock::now();
                double elapsed = std::chrono::duration<double>(now - start_time).count();

                // --- ��ӡͷ��ժҪ ---
                std::printf("\n[%.3fs] +%d bytes (total=%lu) ============\n",
                            elapsed, n, (unsigned long)total_bytes);

                // --- Hex ���д�ӡ + �Ҳ� ASCII ---
                for (int i = 0; i < n; i += 16)
                {
                    // ����ƫ��
                    std::printf("  %04x  ", i);

                    // Hex ��
                    char ascii[17] = {0};
                    int j;
                    for (j = 0; j < 16 && (i + j) < n; j++)
                    {
                        std::printf("%02x ", buf[i + j]);
                        ascii[j] = (buf[i + j] >= 32 && buf[i + j] <= 126)
                                       ? (char)buf[i + j]
                                       : '.';
                    }
                    // ����ո�
                    for (; j < 16; j++)
                    {
                        std::printf("   ");
                        ascii[j] = ' ';
                    }
                    ascii[16] = '\0';
                    std::printf(" |%s|\n", ascii);
                }
                std::fflush(stdout);
            }
            // else: no data, just loop (VTIME ensures we don't busy-spin)
        }
        std::printf("\n[Uart::_dump_loop] Dump stopped. total_bytes=%lu\n",
                    (unsigned long)total_bytes);
    }

    // ============================================================
    //  ԭ��ռλ����ѭ�������ֲ��䣩
    // ============================================================
    void _receive_loop()
    {
        while (running_.load() && !dump_mode_.load())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    std::string port_name_;
    bool is_open_;
    std::thread thread_rec_;
    std::atomic<bool> running_;
    int fd_;                      // �����ļ�������
    std::atomic<bool> dump_mode_; // �Ƿ���ת��ģʽ
    uint8_t rx_buf_[256];
    size_t rx_len_ = 0;

    bool _parse_speed_frame(float &speed)
    {
        while (rx_len_ >= 3)
        {
            if (rx_buf_[0] != USB_FRAME_HEAD)
            {
                _drop_rx_bytes(1);
                continue;
            }

            const uint8_t frame_len = rx_buf_[2];
            if (frame_len < USB_FRAME_LENMIN || frame_len > USB_FRAME_LENMAX)
            {
                _drop_rx_bytes(1);
                continue;
            }

            if (rx_len_ < frame_len)
            {
                return false;
            }

            uint8_t check = 0;
            for (uint8_t i = 0; i < (uint8_t)(frame_len - 1); ++i)
            {
                check = (uint8_t)(check + rx_buf_[i]);
            }
            if (check != rx_buf_[frame_len - 1])
            {
                _drop_rx_bytes(1);
                continue;
            }

            if (rx_buf_[1] == USB_ADDR_CARCTRL && frame_len >= 8)
            {
                union
                {
                    float f;
                    uint8_t b[4];
                } u;
                u.b[0] = rx_buf_[3];
                u.b[1] = rx_buf_[4];
                u.b[2] = rx_buf_[5];
                u.b[3] = rx_buf_[6];
                speed = u.f;
                _drop_rx_bytes(frame_len);
                return true;
            }

            _drop_rx_bytes(frame_len);
        }

        return false;
    }

    void _drop_rx_bytes(size_t n)
    {
        if (n >= rx_len_)
        {
            rx_len_ = 0;
            return;
        }
        std::memmove(rx_buf_, rx_buf_ + n, rx_len_ - n);
        rx_len_ -= n;
    }
};
