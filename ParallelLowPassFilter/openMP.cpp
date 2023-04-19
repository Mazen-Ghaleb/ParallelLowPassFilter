#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/utils/logger.hpp>

#include <omp.h>

using namespace cv;
using namespace std;

Mat openMPLowPassFilter(Mat image, int kernal_size) {
    // Beginning of parallel region
#pragma omp parallel
    {
        printf("Hello World from thread %d\n",
            omp_get_thread_num());
    }
    // Ending of parallel region
    return image;
}

int main2(int argc, char** argv) {
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
    imshow("New image", openMPLowPassFilter(image, kernal_size));

    auto end_time = chrono::high_resolution_clock::now();
    auto elapsed_time = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);

    printf("Elapsed time: %lld milliseconds", elapsed_time.count());

    waitKey(0);

    return 0;
}