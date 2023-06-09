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
    int leftOver = paddedImage.rows % world_size;

    int prevRank = world_rank - 1;
    int nextRank = world_rank + 1;

    // Perform convolution on the local image block using the filter kernel
    Mat aboveRows = Mat::zeros(paddingSize, localWidth, CV_8UC1);
    Mat belowRows = Mat::zeros(paddingSize, localWidth, CV_8UC1);

    // Scatter the input image to all processes
    int* sendcounts = new int[world_size];
    int* displs = new int[world_size];

    // Calculate the sendcounts and displacements
    for (int i = 0; i < world_size; i++) {
        sendcounts[i] = localHeight * localWidth;
        displs[i] = i * localHeight * localWidth;

        if (i < leftOver) {
            sendcounts[i] += localWidth;
            displs[i] +=  (i * localWidth);
        }
        else {
            displs[i] += (leftOver * localWidth);
        }
    }

    // Fix the local height according to leftover distribution
    if (world_rank < leftOver) {
        localHeight += 1;
    }

    // Allocate memory for the local image block and the output block
    Mat localImage(localHeight, localWidth, inputImage.type());
    Mat localOutputImage(localHeight, localWidth, inputImage.type());

    // Debugging print
    //if (world_rank == 0) {
    //    for (int i = 0; i < world_size; i++) {
    //        printf("%d \n", displs[i]);
    //        fflush(stdout);
    //    }
    //}
    //printf("Image size: [%d x %d]\n", localImage.rows, localImage.cols);
    //fflush(stdout);


    // Scatter the input image to all processes
    MPI_Scatterv(paddedImage.data, sendcounts, displs, MPI_UNSIGNED_CHAR, localImage.data, localHeight * localWidth, MPI_UNSIGNED_CHAR, collector, MPI_COMM_WORLD);

    // Send the top and bottom rows to the previous and next processes
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

    // Perform convolution on the local image block using the filter kernel
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

    // Initialize the output image on the root process
    Mat outputImage;
    if (world_rank == collector) {
        outputImage = Mat::zeros(paddedImage.size(), paddedImage.type());
    }

    // Gather the filtered blocks from all processes to the root process
    MPI_Gatherv(localOutputImage.data, localHeight * localWidth, MPI_UNSIGNED_CHAR, outputImage.data, sendcounts, displs, MPI_UNSIGNED_CHAR, collector, MPI_COMM_WORLD);

    // Free the memory
    delete[] sendcounts;
    delete[] displs;

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
        // Start the timer
        auto start_time = chrono::high_resolution_clock::now();

        // Perform convolution on the input image using the filter kernel
        Mat outputImage = MPILowPassFilter(image, kernal_size, world_size, world_rank, collector);

        if (world_rank == collector) {
            // Print the elapsed time in milliseconds
            auto end_time = chrono::high_resolution_clock::now();
            auto elapsed_time = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
            printf("MPI Elapsed time: %lld milliseconds\n", elapsed_time.count());
            fflush(stdout);

            // Display the input and output images
            imshow("Original image", image);
            imshow("MPI Output image", outputImage);
            imwrite("mpiImage.png", outputImage);

            // Wait for the user to press any key
            if (waitFlag) {
                waitKey(0);
            }
        }

        // Return the output image
        return outputImage;
    }

    Mat process(const Mat& image, const int kernal_size, const int world_size, const int world_rank, const int collector) {
        // Perform convolution on the input image
        return process(image, kernal_size, world_size, world_rank, collector, true);
    }
}