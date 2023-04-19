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

Mat seqLowPassFilter(const Mat& inputImage, int kernelSize)
{
    // Define the filter kernel
    //Mat kernel = Mat::ones(kernelSize, kernelSize, CV_32F) / (float)(kernelSize * kernelSize);

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
                    /*float kernelValue = kernel.at<float>(k + paddingSize, l + paddingSize);
                    sum += pixelValue * kernelValue;*/
                    sum += pixelValue * filter_value;
                }
            }
            outputImage.at<uchar>(i - paddingSize, j - paddingSize) = sum;
        }
    }

    return outputImage;
}

int main3(int argc, char** argv) {
    // Set OPENCV LOG level to ERROR
    utils::logging::setLogLevel(utils::logging::LogLevel::LOG_LEVEL_ERROR);

    int kernal_size;
    cout << "Enter the kernel size: ";
    cin >> kernal_size;

    auto start_time = chrono::high_resolution_clock::now();

    //Mat image = imread("untitled.png", IMREAD_COLOR);
    Mat image = imread("untitled.png", IMREAD_GRAYSCALE);
    if (image.empty()) {
        cout << "Could not read the image" << endl;
        return 1;
    }

    imshow("Original image", image);
    imshow("New image", seqLowPassFilter(image, kernal_size));

    auto end_time = chrono::high_resolution_clock::now();
    auto elapsed_time = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);

    printf("Elapsed time: %lld milliseconds", elapsed_time.count());

    waitKey(0);

    return 0;
}