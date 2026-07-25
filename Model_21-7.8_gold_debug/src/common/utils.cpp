#include "utils.hpp"

using namespace cv;
using namespace std;

int clip(int x, int low, int up)
{
    return x > up ? up : x < low ? low : x;
}

float clipf(float x, float low, float up)
{
    return x > up ? up : x < low ? low : x;
}

void swap(int *a, int *b)
{
    int temp = *a;
    *a = *b;
    *b = temp;
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