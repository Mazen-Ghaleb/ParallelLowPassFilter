#pragma once
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <mpi.h>

using namespace cv;
using namespace std;

Mat MPILowPassFilter(const Mat& inputImage, const int kernelSize, const int world_size, const int world_rank, const int collector);

namespace mpi
{
	Mat process(const Mat& image, const int kernal_size, const int world_size, const int world_rank, const int collector, const bool waitFlag);
	Mat process(const Mat& image, const int kernal_size, const int world_size, const int world_rank, const int collector);
}