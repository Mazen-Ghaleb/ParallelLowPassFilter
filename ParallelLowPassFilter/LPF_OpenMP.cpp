#include "LPF_OpenMP.h"

Mat openMPLowPassFilter(const Mat& inputImage, const int kernelSize) {
    // Define the filter kernel
    float filter_value = (1 / (float)(kernelSize * kernelSize));

    // Perform zero padding on the input image
    int paddingSize = kernelSize / 2;
    Mat paddedImage;
    copyMakeBorder(inputImage, paddedImage, paddingSize, paddingSize, paddingSize, paddingSize, BORDER_CONSTANT, Scalar(0));

    // Create an output image with the same size as the input image
    Mat outputImage(inputImage.size(), inputImage.type());

    // Perform convolution on the padded image using the filter kernel
#pragma omp parallel for collapse(2)
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

namespace openmp
{
    Mat process(const Mat& image, const int kernal_size, const bool waitFlag) {
        auto start_time = chrono::high_resolution_clock::now();
        Mat outputImage = openMPLowPassFilter(image, kernal_size);
        auto end_time = chrono::high_resolution_clock::now();
        auto elapsed_time = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
        printf("OpenMP Elapsed time: %lld milliseconds\n", elapsed_time.count());
        fflush(stdout);

        imshow("Original image", image);
        imshow("OpenMP Output image", outputImage);
        imwrite("OpenMPImage.png", outputImage);

        if (waitFlag) {
            waitKey(0);
        }

        return outputImage;
    }

    Mat process(const Mat& image, const int kernal_size) {
        return process(image, kernal_size, true);
    }
}