#include "LPF_OpenMP.h"

Mat openMPLowPassFilter(const Mat& inputImage, const int kernelSize, const int num_of_threads) {
    // Define the filter kernel
    float filter_value = (1 / (float)(kernelSize * kernelSize));

    // Perform zero padding on the input image
    int paddingSize = kernelSize / 2;
    Mat paddedImage;
    copyMakeBorder(inputImage, paddedImage, paddingSize, paddingSize, paddingSize, paddingSize, BORDER_CONSTANT, Scalar(0));

    // Create an output image with the same size as the input image
    Mat outputImage(inputImage.size(), inputImage.type());

    //// Print the number of threads
    //printf("Number of threads: %d\n", num_of_threads);

    // Perform convolution on the padded image using the filter kernel
#pragma omp parallel for collapse(2) num_threads(num_of_threads)
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
    Mat process(const Mat& image, const int kernal_size, const int num_of_threads, const bool waitFlag) {
        // Start the timer
        auto start_time = chrono::high_resolution_clock::now();

        // Perform convolution on the input image using the filter kernel
        Mat outputImage = openMPLowPassFilter(image, kernal_size, num_of_threads);

        // Stop the timer
        auto end_time = chrono::high_resolution_clock::now();

        // Print the elapsed time in milliseconds
        auto elapsed_time = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
        printf("OpenMP Elapsed time: %lld milliseconds\n", elapsed_time.count());
        fflush(stdout);

        // Display the input and output images
        imshow("Original image", image);
        imshow("OpenMP Output image", outputImage);
        imwrite("OpenMPImage.png", outputImage);

        // Wait for the user to press any key
        if (waitFlag) {
            waitKey(0);
        }

        // Return the output image
        return outputImage;
    }

    Mat process(const Mat& image, const int kernal_size, const int num_of_threads) {
        // Perform convolution on the input image
        return process(image, kernal_size, num_of_threads, true);
    }
}