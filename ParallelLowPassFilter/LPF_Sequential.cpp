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
        // Start the timer
        auto start_time = chrono::high_resolution_clock::now();

        // Perform convolution on the input image using the filter kernel
        Mat outputImage = seqLowPassFilter(image, kernal_size);

        // Stop the timer
        auto end_time = chrono::high_resolution_clock::now();

        // Print the elapsed time in milliseconds
        auto elapsed_time = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
        printf("Sequential Elapsed time: %lld milliseconds\n", elapsed_time.count());
        fflush(stdout);

        // Display the input and output images
        imshow("Original image", image);
        imshow("Sequential Output image", outputImage);
        imwrite("seqImage.png", outputImage);

        // Wait for the user to press any key
        if (waitFlag) {
            waitKey(0);
        }

        // Return the output image
        return outputImage;
    }

    Mat process(const Mat& image, const int kernal_size) {
        // Perform convolution on the input image
        return process(image, kernal_size, true);
    }
}