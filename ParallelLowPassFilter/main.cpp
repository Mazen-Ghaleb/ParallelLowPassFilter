#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <chrono>

#include "LPF_Sequential.h"
#include "LPF_openMP.h"
#include "LPF_MPI.h"

using namespace cv;
using namespace std;

void compareImage(String image1, String image2, String windowName) { // Compare two images using Paths
    Mat a;
    Mat b = imread(image1, IMREAD_COLOR);
    Mat c = imread(image2, IMREAD_COLOR);

    absdiff(b, c, a);
    imshow(windowName, a);
}

void compareImage(const Mat& image1, const Mat& image2, String windowName) { // Compare two images using Matrices
    Mat a;

    absdiff(image1, image2, a);
    imshow(windowName, a);
}

void compareImageMSE(const Mat& image1, const Mat& image2, String windowName) { // Compare two images using Matrices
    Mat diff;

    absdiff(image1, image2, diff);
    double mse = mean(diff.mul(diff))[0]; // compute mean squared error
    if (mse > 0) { // if there is a difference
        cout <<"At "<< windowName << ", Images are different (MSE = " << mse << ")" << endl;
    }
    else {
        cout << "At " << windowName <<", Images are the same" << endl;
    }
}

void process(const Mat& image, const int& kernal_size, const int& world_size, const int& world_rank, const int& collector) {

    Mat MPI_outputImage = mpi::process(image, kernal_size, world_size, world_rank, collector, false);

    if (world_rank == collector) {
        Mat Seq_outputImage =  sequential::process(image, kernal_size, false);
        Mat openMP_outputImage = openmp::process(image, kernal_size, false);

        compareImageMSE(Seq_outputImage, openMP_outputImage, "Seq vs openMP");
        compareImageMSE(Seq_outputImage, MPI_outputImage, "Seq vs MPI");
        compareImageMSE(openMP_outputImage, MPI_outputImage, "openMP vs MPI");

        waitKey(0);
    }
}

int main(int argc, char** argv) {

    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    // Set OPENCV LOG level to ERROR
    utils::logging::setLogLevel(utils::logging::LogLevel::LOG_LEVEL_ERROR);

    Mat image = imread("untitled.png", IMREAD_GRAYSCALE);
    if (image.empty()) {
        cout << "Could not read the image" << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    const int collector = 0; // The processor that will send the message

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int method = 4; // The method that will be used to process the image with default value of 4 (all methods)
    int kernal_size = 3; // The size of the kernel with default value of 3

    if (world_rank == collector) {
        cout << "Remember to press ESC on window to continue" << endl << endl;
    }

    while (true) {
        if (world_rank == collector) {
        destroyAllWindows();
        cout << "Enter the number for the method you want to use: " << endl;
        cout << "1- Sequential" << endl;
        cout << "2- OpenMP" << endl;
        cout << "3- MPI" << endl;
        cout << "4- All" << endl;
        cout << "5- Terminate" << endl;
        cout << endl << "Method: ";
        cin >> method;
        }

        // Broadcast the selected method to all other processes
        MPI_Bcast(&method, 1, MPI_INT, collector, MPI_COMM_WORLD);

        switch (method) {
        case 5:
            goto exit_loop;
            break;
        case 1:
        case 2:
        case 3:
        case 4:
            if (world_rank == collector) {
                do {
                    cout << "Enter the kernel size: ";
                    cin >> kernal_size;

                    if (kernal_size % 2 == 0 || kernal_size < 0) {
                        cout << "The kernel size must be a positive odd number" << endl;
                    }
                } while (kernal_size % 2 == 0 || kernal_size < 0);
            }
            switch (method) {
            case 1:
                if (world_rank == collector) {
                    sequential::process(image, kernal_size);
                }
                break;
            case 2:
                if (world_rank == collector) {
                    openmp::process(image, kernal_size);
                }
                break;
            case 3:
                MPI_Bcast(&kernal_size, 1, MPI_INT, collector, MPI_COMM_WORLD);
                mpi::process(image, kernal_size, world_size, world_rank, collector);
                break;
            case 4:
                MPI_Bcast(&kernal_size, 1, MPI_INT, collector, MPI_COMM_WORLD);
                process(image, kernal_size, world_size, world_rank, collector);
                break;
            }
            break;
    default:
        if (world_rank == collector) {
            cout << "The method number must be a number from 1 to 5" << endl;
        }
		break;
        }

        if (world_rank == collector) {
            cout << endl;
        }
    }
    exit_loop:;
    
    // Finalize the MPI environment.
    MPI_Finalize();
    return 0;
}