#include "LPF_MPI.h"

Mat MPILowPassFilter(const Mat& inputImage, const int kernelSize, const int world_size, const int world_rank, const int collector) {
    // Define the filter kernel
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

    int prevRank = world_rank - 1;
    int nextRank = world_rank + 1;

    // Perform convolution on the local image block using the filter kernel
    Mat aboveRows = Mat::zeros(paddingSize, localWidth, CV_8UC1);
    Mat belowRows = Mat::zeros(paddingSize, localWidth, CV_8UC1);

    if (world_size > 1) {
        if (prevRank >= 0) {
            MPI_Request request;
            MPI_Isend(localImage.rowRange(0, paddingSize).data, paddingSize * localWidth, MPI_UNSIGNED_CHAR, prevRank, 0, MPI_COMM_WORLD, &request);
            MPI_Recv(aboveRows.data, paddingSize * localWidth, MPI_UNSIGNED_CHAR, prevRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        if (nextRank <= world_size - 1) {
            MPI_Request request2;
            MPI_Isend(localImage.rowRange(localHeight - paddingSize, localHeight).data, paddingSize * localWidth, MPI_UNSIGNED_CHAR, nextRank, 0, MPI_COMM_WORLD, &request2);
            MPI_Recv(belowRows.data, paddingSize * localWidth, MPI_UNSIGNED_CHAR, nextRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    for (int i = 0; i < localHeight; i++) {
        for (int j = paddingSize; j < localWidth - paddingSize; j++) {
            float sum = 0.0;
            for (int k = -paddingSize; k <= paddingSize; k++) {
                for (int l = -paddingSize; l <= paddingSize; l++) {
                    if ((i + k >= 0) && (i + k < localHeight) && (j + l >= 0) && (j + l < localWidth)) {
                        float pixelValue = localImage.at<uchar>(i + k, j + l);
                        sum += pixelValue * filter_value;
                    }
                    else if (i + k < 0) {
                        int index = ((i + k) + paddingSize) % paddingSize;
                        sum += aboveRows.at<uchar>(index, j + l) * filter_value; // First row - access above block
                    }
                    else if (i + k >= localHeight) {
                        int index = ((i + k - localHeight) + paddingSize) % paddingSize;
                        sum += belowRows.at<uchar>(index, j + l) * filter_value;// Last row - access below block
                    }
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

namespace mpi
{
    Mat process(const Mat& image, const int kernal_size, const int world_size, const int world_rank, const int collector, const bool waitFlag) {
        auto start_time = chrono::high_resolution_clock::now();
        Mat outputImage = MPILowPassFilter(image, kernal_size, world_size, world_rank, collector);

        if (world_rank == collector) {
            auto end_time = chrono::high_resolution_clock::now();
            auto elapsed_time = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
            printf("MPI Elapsed time: %lld milliseconds\n", elapsed_time.count());
            fflush(stdout);

            imshow("Original image", image);
            imshow("MPI Output image", outputImage);
            imwrite("mpiImage.png", outputImage);

            if (waitFlag) {
                waitKey(0);
            }
        }
        return outputImage;
    }

    Mat process(const Mat& image, const int kernal_size, const int world_size, const int world_rank, const int collector) {
        return process(image, kernal_size, world_size, world_rank, collector, true);
    }
}