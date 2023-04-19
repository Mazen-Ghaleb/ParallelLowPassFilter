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

Mat MPILowPassFilter(const Mat& inputImage, int kernelSize, int world_size, int world_rank, int collector) {

    // Define the filter kernel
   //Mat kernel = Mat::ones(kernelSize, kernelSize, CV_32F) / (float)(kernelSize * kernelSize);
    float filter_value = (1 / (float)(kernelSize * kernelSize));

    // Perform zero padding on the input image
    int paddingSize = kernelSize / 2;
    Mat paddedImage;
    copyMakeBorder(inputImage, paddedImage, paddingSize, paddingSize, paddingSize, paddingSize, BORDER_CONSTANT, Scalar(0));

    // Calculate the local dimensions and offset for each process
    int localHeight = paddedImage.rows / world_size;
    int localWidth = paddedImage.cols;
    int localOffset = world_rank * localHeight;

    // Allocate memory for the local image block and the output block
    Mat localImage(localHeight, localWidth, inputImage.type());
    Mat localOutputImage(localHeight, localWidth, inputImage.type());

    // Scatter the input image to all processes
    MPI_Scatter(paddedImage.data, localHeight * localWidth, MPI_UNSIGNED_CHAR, localImage.data, localHeight * localWidth, MPI_UNSIGNED_CHAR, collector, MPI_COMM_WORLD);

    // Perform convolution on the local image block using the filter kernel
    for (int i = paddingSize; i < localHeight - paddingSize; i++) {
        for (int j = paddingSize; j < localWidth - paddingSize; j++) {
            float sum = 0.0;
            for (int k = -paddingSize; k <= paddingSize; k++) {
                for (int l = -paddingSize; l <= paddingSize; l++) {
                    float pixelValue = localImage.at<uchar>(i + k, j + l);
                    /*float kernelValue = kernel.at<float>(k + paddingSize, l + paddingSize);
                    sum += pixelValue * kernelValue;*/
                    sum += pixelValue * filter_value;

                }
            }
            localOutputImage.at<uchar>(i, j) = sum;
        }
    }

    // Gather the filtered blocks from all processes to the root process
    Mat outputImage;
    if (world_rank == collector) {
        outputImage = Mat::zeros(paddedImage.size(), paddedImage.type());
    }
    MPI_Gather(localOutputImage.data, localHeight * localWidth, MPI_UNSIGNED_CHAR, outputImage.data, localHeight * localWidth, MPI_UNSIGNED_CHAR, collector, MPI_COMM_WORLD);

    // Crop the output image to remove the padding and return it
    if (world_rank == collector) {
        Rect cropRegion(paddingSize, paddingSize, inputImage.cols, inputImage.rows);
        return outputImage(cropRegion);
    }
    else {
        return Mat();
    }
}

int main(int argc, char** argv) {
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    // Set OPENCV LOG level to ERROR
    utils::logging::setLogLevel(utils::logging::LogLevel::LOG_LEVEL_ERROR);

    int collector = 0; // The processor that will send the message

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    
    int kernal_size = 3; // default value

    if (world_rank == collector) {
        cout << "Enter the kernel size: ";
        cin >> kernal_size;
    }
    // Broadcast the kernel size to all processes
    MPI_Bcast(&kernal_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    //Mat image = imread("untitled.png", IMREAD_COLOR);
    Mat image = imread("untitled.png", IMREAD_GRAYSCALE);
    if (image.empty()) {
        cout << "Could not read the image" << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    auto start_time = chrono::high_resolution_clock::now();

    Mat filtered_image = MPILowPassFilter(image, kernal_size, world_size, world_rank, collector);
    auto end_time = chrono::high_resolution_clock::now();
    auto elapsed_time = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);

    if (world_rank == collector) {
        printf("Elapsed time: %lld milliseconds", elapsed_time.count());
        fflush(stdout);
        imshow("Original image", image);
        imshow("New image", filtered_image);
        waitKey(0);
    }

    // Finalize the MPI environment.
    MPI_Finalize();
    return 0;
}