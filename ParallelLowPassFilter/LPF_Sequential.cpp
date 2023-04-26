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

Mat seqLowPassFilter(const Mat& inputImage, const int kernelSize)
{
    float filter_value = (1 / (float)(kernelSize * kernelSize));

    // Perform zero padding on the input image
    int paddingSize = kernelSize / 2;
    Mat paddedImage;
    copyMakeBorder(inputImage, paddedImage, paddingSize, paddingSize, paddingSize, paddingSize, BORDER_CONSTANT, Scalar(0));

    // Create an output image with the same size as the input image
    Mat outputImage(inputImage.size(), inputImage.type());

    // Perform convolution on the padded image using the filter kernel
    for (int i = paddingSize; i < paddedImage.rows - paddingSize; i++) {
        for (int j = paddingSize; j < paddedImage.cols - paddingSize; j++) {
            float sum = 0.0;
            for (int k = -paddingSize; k <= paddingSize; k++) {
                for (int l = -paddingSize; l <= paddingSize; l++) {
                    float pixelValue = paddedImage.at<uchar>(i + k, j + l);
                    sum += pixelValue * filter_value;
                }
            }
            outputImage.at<uchar>(i - paddingSize, j - paddingSize) = sum;
        }
    }

    return outputImage;
}

namespace sequential
{
    Mat process(const Mat& image, const int kernal_size, const bool waitFlag) {
        auto start_time = chrono::high_resolution_clock::now();
        Mat outputImage = seqLowPassFilter(image, kernal_size);
        auto end_time = chrono::high_resolution_clock::now();
        auto elapsed_time = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
        printf("Sequential Elapsed time: %lld milliseconds\n", elapsed_time.count());
        fflush(stdout);

        imshow("Original image", image);
        imshow("Sequential Output image", outputImage);
        imwrite("seqImage.png", outputImage);

        if (waitFlag) {
            waitKey(0);
        }

        return outputImage;
    }

    Mat process(const Mat& image, const int kernal_size) {
        return process(image, kernal_size, true);
    }
}