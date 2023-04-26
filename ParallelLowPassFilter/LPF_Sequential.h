#pragma once
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <chrono>

using namespace cv;
using namespace std;

Mat seqLowPassFilter(const Mat& inputImage, const int kernelSize);

namespace sequential
{
	Mat process(const Mat& image, const int kernal_size, const bool waitFlag);
	Mat process(const Mat& image, const int kernal_size);
}

