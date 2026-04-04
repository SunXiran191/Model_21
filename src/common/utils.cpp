#include "utils.hpp"

using namespace cv;
using namespace std;

int clamp(int val, int min_val, int max_val) {
    return max(min_val, min(val, max_val));
}

int factorial(int x)
{
    int f = 1;
    for (int i = 1; i <= x; i++)
    {
        f *= i;
    }
    return f;
}