#include <atomic>
#include <iostream>

// Global stop flag: 0 means keep running, 1 means stop requested.
std::atomic<int> stop_flag(0);

// Blocking wait: once keyboard input contains '0', set the global flag to 1.
void wait_for_zero_and_set_flag()
{
    char ch = 0;
    while (std::cin.get(ch))
    {
        if (ch == '0')
        {
            stop_flag.store(1, std::memory_order_relaxed);
            break;
        }
    }
}
