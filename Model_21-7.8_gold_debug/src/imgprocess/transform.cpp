#include "transform.hpp"

#include <cmath>
#include <iostream>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

// double change_un_Mat[3][3] = {
//     {-6.38545023e-01, -2.30721047e+00, 5.40111168e+02},
//     {1.17822034e-01, -4.18222786e+00, 7.31151604e+02},
//     {2.57787191e-04, -7.51702162e-03, 1.00000000e+00}};

// double Re_change_un_Mat[3][3] = {
//     {-1.30105660, 0.71834101, -131.391217},
//     {0.03661902, -0.19873202, 145.291016},
//     {0.00033490, -0.00552113, 1.000000}};

double change_un_Mat[3][3] = {
    {-5.24901912e-01, -1.53674139e+00, 4.88523382e+02},
    {-1.96112302e-02, -2.57146117e+00, 6.40316978e+02},
    {-4.26331091e-05, -4.68063546e-03, 1.00000000e+00}};

double Re_change_un_Mat[3][3] = {
    {-1.93192603e+00, 3.40359911e+00, -1.23559126e+03},
    {3.48933545e-02, 2.28798908e+00, -1.48208447e+03},
    {8.09590590e-05, 1.08543488e-02, -5.98977423e+00}};

cv::Point2f transf(float i, float j)
{
    double w = change_un_Mat[2][0] * i + change_un_Mat[2][1] * j + change_un_Mat[2][2];
    if (std::abs(w) < 1e-6)
        return cv::Point2f(0.0f, 0.0f);

    double nx = (change_un_Mat[0][0] * i + change_un_Mat[0][1] * j + change_un_Mat[0][2]) / w;
    double ny = (change_un_Mat[1][0] * i + change_un_Mat[1][1] * j + change_un_Mat[1][2]) / w;

    return cv::Point2f(static_cast<float>(nx), static_cast<float>(ny));
}

cv::Point2f reverse_transf(float i, float j)
{
    double w = Re_change_un_Mat[2][0] * i + Re_change_un_Mat[2][1] * j + Re_change_un_Mat[2][2];
    if (std::abs(w) < 1e-6)
        return cv::Point2f(0.0f, 0.0f);

    double nx = (Re_change_un_Mat[0][0] * i + Re_change_un_Mat[0][1] * j + Re_change_un_Mat[0][2]) / w;
    double ny = (Re_change_un_Mat[1][0] * i + Re_change_un_Mat[1][1] * j + Re_change_un_Mat[1][2]) / w;

    return cv::Point2f(static_cast<float>(nx), static_cast<float>(ny));
}
