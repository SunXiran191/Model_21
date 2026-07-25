/**
 * @file uart_dump.cpp
 * @brief ´®¿ÚÊý¾Ý¼àÌý¹¤¾ß ¡ª ÓÃÓÚ·ÖÎöÏÂÎ»»úÊý¾Ý¸ñÊ½
 *
 * ±àÒë:
 *   g++ -std=c++11 -pthread uart_dump.cpp -o uart_dump
 *
 * ÔËÐÐ:
 *   ./uart_dump /dev/ttyUSB0 115200
 *   (°´ Ctrl+C ÍË³ö)
 */

#include "../res/include/uartReceive.hpp"
#include <csignal>
#include <cstdlib>
#include <cstring>

static Uart *g_uart = nullptr;

void signal_handler(int)
{
    std::cout << "\nCaught signal, stopping dump..." << std::endl;
    if (g_uart)
    {
        g_uart->stop_dump();
    }
}

void print_usage(const char *prog)
{
    std::cerr << "Usage: " << prog << " <port> [baud_rate]" << std::endl;
    std::cerr << "  port      : ´®¿ÚÉè±¸Â·¾¶, e.g. /dev/ttyUSB0" << std::endl;
    std::cerr << "  baud_rate : ²¨ÌØÂÊ (¿ÉÑ¡, Ä¬ÈÏ 115200)" << std::endl;
    std::cerr << "              Ö§³Ö: 9600, 19200, 38400, 57600, 115200" << std::endl;
}

speed_t parse_baud(int rate)
{
    switch (rate)
    {
    case 9600:
        return B9600;
    case 19200:
        return B19200;
    case 38400:
        return B38400;
    case 57600:
        return B57600;
    case 115200:
        return B115200;
    default:
        std::cerr << "Unsupported baud rate: " << rate << ", using 115200" << std::endl;
        return B115200;
    }
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        print_usage(argv[0]);
        return 1;
    }

    const char *port = argv[1];
    speed_t baud = B115200;
    if (argc >= 3)
    {
        baud = parse_baud(std::atoi(argv[2]));
    }

    signal(SIGINT, signal_handler);

    Uart uart(port);
    g_uart = &uart;

    std::cout << "============================================" << std::endl;
    std::cout << "  UART Data Dump Tool" << std::endl;
    std::cout << "  Port : " << port << std::endl;
    std::cout << "  Baud : " << (argc >= 3 ? argv[2] : "115200") << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "Press Ctrl+C to stop." << std::endl;
    std::cout << std::endl;

    if (uart.open_dump(baud) != 0)
    {
        std::cerr << "Failed to open port!" << std::endl;
        return 1;
    }

    uart.start_dump();

    // Ö÷Ïß³ÌµÈ´ý£¬Ö±µ½ÊÕµ½Í£Ö¹ÐÅºÅ
    while (uart.is_dumping())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    uart.close();
    std::cout << "Exited cleanly." << std::endl;
    return 0;
}
