#define _USE_MATH_DEFINES

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <math.h> 
#include <vector>
#include "canny.h"


using namespace cv;
using namespace std;


int main() {

    namedWindow("Webcam", WINDOW_NORMAL);
    int low = 0;
    int high = 50;
    int fps = 20;

    Mat frame;

    VideoCapture video(0);
    if (!video.isOpened()) {
        return -1;
    }
    while (video.read(frame))
    {
        Mat img;
        resize(frame, img, cv::Size(), 0.4, 0.4);
        cvtColor(img, img, COLOR_BGR2GRAY);
        Mat canny = img.clone();
        canny = Cany(img, high, low);
        imshow("Webcam", canny);
        if (waitKey(1000 / fps) >= 0)
            break;
    }
    waitKey(0);
    return 0;
 


}