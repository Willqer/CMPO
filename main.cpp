//
//       *** PROJECT DONE BY GROUP F ***
// BIENIEK, KULESZA, KULI�SKI, PI�TKA, ZAWADZKA
//                31.01.2021
//

#define _USE_MATH_DEFINES
#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>
#include <vector>

using namespace cv;
using namespace std;
//defining Array and Matrix "variables" (containers)
typedef vector<double> Array;
typedef vector<Array> Matrix;
//declarations of functions that will be used
Mat gaussian_filter(Mat);
Mat customErosion(Mat, int,int);
Mat threshold(int, Mat);
void setLabel(Mat&, const string, vector<Point>&);

int main()
{
    //declaration of Mat object for video frames
    Mat frame;
    //capturing video from webcam
    VideoCapture video(0);
    //checking if capturing of video works
    if (!video.isOpened()) {
        return -1;
    }
    //main loop of programme which continues while frames are taken from webcam video to 'frame' variable
    while (video.read(frame))
    {
        //declaration of Mat object for resized image (frame)
        Mat img;
        //resizing frame (optimization) and putting it to Mat img
        resize(frame, img, Size(), 0.5, 0.5);
        //color space conversion from BGR to Grayscale
        cvtColor(img, img, COLOR_BGR2GRAY);
        //performing Gauss Filtration (own implementation) on image to reduce noises
        Mat gauss = gaussian_filter(img);
        //performing quasi-erosion (own algorithm) on image to reduce noises
        Mat eroded = customErosion(gauss, 3, 4);
        //performing local threshold (own algorithm) on image to binaries it
        //this function extracts from image only contours of shapes in input
        Mat thresh = threshold(5, eroded);
        //declaration of needed variables (vectors)
        vector <vector<Point> > contours;
        vector <Vec4i> hierarchy;
        vector <Point> approx;
        //RANDOM COLOR GENERATOR variable (some random number need further)
        RNG rng;
        //finding contours in image which will be held in 'contours' variable
        findContours(thresh, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
        //Mat object for drawing color contours
        Mat drawing = Mat::zeros(thresh.size(), CV_8UC3);
        //loop for drawing founded contours with random colors
        for (int i = 0; i < contours.size(); i++)
        {
            //container for random colors
            Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
            //drawing of contours
            drawContours(drawing, contours, i, color, 2, LINE_8, hierarchy, 0);
        }
        //loop for finding shapes in our contours
        for (int i = 0; i < contours.size(); i++)
        {
            //approximation of contours shapes to simpler ones with Douglas-Peucker algorithm
            approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true) * 0.035, true);
            //omitting the small contours
            if (fabs(contourArea(contours[i])) < 150 || !isContourConvex(approx)) continue;
            //putting number of vertices to new variable
            int vtc = approx.size();
            //checking the number of vertices for different shapes
            if (vtc == 4)
                //rectangle
                setLabel(drawing, "RECT", contours[i]);
            else if (vtc == 5)
                //pentagon
                setLabel(drawing, "PENTA", contours[i]);
            else if (vtc >= 6)
                //star
                setLabel(drawing, "STAR", contours[i]);
        }
        //normalizing display window
        namedWindow("Webcam", WINDOW_NORMAL);
        //showing of drawn contours and founded shapes
        imshow("Webcam", drawing);
        //going to another loop pass after some time
        if (waitKey(1000 / 20) >= 0)
            break;
    }
    //waiting for user before closing the app
    waitKey(0);

    return 0;
}

//definition of threshold function
Mat threshold(int kernelSize, Mat img)
{
    //declaration of needed variables
    int mean, max_val = 255;
    //capturing the size of Mat being binarized
    Size imgSize = img.size();
    //dividing size on rows and columns
    int imgHeight = imgSize.height;
    int imgWidth = imgSize.width;
    //making output Mat object with input parameters
    Mat resultImg = Mat(imgHeight, imgWidth, img.type());
    //moving through every single pixel of matrix (Mat obj)
    for (int i = 0; i < imgHeight; i++) {
        for (int j = 0; j < imgWidth; j++) {
            //border issue - putting low binarization value to border pixels, size of border depends on kernel size
            //in this 'if' we check if our point (pixel) is a border one
            if ((i < ((kernelSize - 1) / 2)) || (i >= imgHeight - ((kernelSize - 1) / 2)) || (j < ((kernelSize - 1) / 2)) || (j >= imgHeight - ((kernelSize - 1) / 2)))
            {
                //low-bin value for border pixels
                resultImg.at<uchar>(i, j) = 0;
            }
            else
            {
                //declaration of needed variable
                int val = 0;
                //moving through neighbourhood (kernel) of pixel (based on chosen size (diameter))
                for (int k = i; k < i + kernelSize; k++) {
                    for (int l = j; l < j + kernelSize; l++) {
                        //taking value from point (pixel) pointed by coordinates
                        val += img.at<uchar>(k - ((kernelSize - 1) / 2), l - ((kernelSize - 1) / 2));
                    }
                }
                //calculating the kernel mean
                mean = val / (kernelSize * kernelSize);
                //performing of threshold if condition - higher value than mean + SOME BIAS - is fulfilled
                //The BIAS VALUE regulates the level for threshold and have significant impact on binarized output
                if (img.at<uchar>(i, j) > (mean + 5))
                {
                    //giving the the high binarization value to output Mat point
                    resultImg.at<uchar>(i, j) = max_val;
                }
                else
                {
                    //giving the low binarization value to output Mat point
                    resultImg.at<uchar>(i, j) = 0;
                }
            }
        }
    }
    return resultImg;
}

//definition of erosion function
Mat customErosion(Mat img, int kernelSize, int erosion_threshold) {
    //erosion threshold is minimal ammount of pixels around currently analised pixel
    //to keep it in erosion functions. Value may need to be changed for bigger kernels

    //variables with original image sizes
    Size imgSize = img.size();
    int imgHeight = imgSize.height;
    int imgWidth = imgSize.width;

    //variables with new image sizes
    int newImgHeight = imgHeight - kernelSize + 1;
    int newImgWidth = imgWidth - kernelSize + 1;

    //output image
    Mat resultImg = Mat(newImgHeight, newImgWidth, img.type());

    //loops going through pixels on original image
    for (int i = 0; i < newImgHeight; i++)
    {
        for (int j = 0; j < newImgWidth; j++)
        {
            //variable used for counting ammount of white ("full") pixels
            int value = 0;
            //loops going through original image pixels included in kernel
            for (int m = i; m < i + kernelSize; m++)
            {
                for (int n = j; n < j + kernelSize; n++)
                {
                    //adding pixel values (effectivelly it allows counting "full" pixels
                    value += img.at<uchar>(m, n);
                }
            }
            //checking ammount of "full" pixels
            if (value <= (erosion_threshold * 255))
            {
                //if pixel dosen't have enough "full" pixels around, then it is "empty"
                resultImg.at<uchar>(i, j) = 0;
            }
            else {
                //if pixel has enough "full pixels around it, then it is "full"
                resultImg.at<uchar>(i, j) = 255;
            }
        }
    }

    return resultImg;
}

//definition of text-typing function
void setLabel(Mat& im, const string label, vector <Point>& contour)
{
    //text and text container parameters
    int fontface = FONT_HERSHEY_SIMPLEX;
    double scale = 0.3;
    int thickness = 1;
    int baseline = 0;
    //gives text's size based on parameters
    Size text = getTextSize(label, fontface, scale, thickness, &baseline);
    //creates rectangle put over shape
    Rect r = boundingRect(contour);

    //definition of point in the middle of rectangle put over shape
    Point pt(r.x + ((r.width - text.width) / 2), r.y + ((r.height + text.height) / 2));

    //creates container for text in the middle of rectangle put over shape (using point pt)
    rectangle(im, pt + Point(0, baseline), pt + Point(text.width, -text.height), CV_RGB(255, 255, 255), FILLED);
    //puts text in created container
    putText(im, label, pt, fontface, scale, CV_RGB(0, 0, 0), thickness, 8);
}

//definition of gaussian-filter function
Mat gaussian_filter(Mat img)
{
    //declaration of needed variables
    int f, g;

    Mat resultImg = Mat(img.size(), img.type());

    // initialising standard deviation to 1.0
    double sigma = 1.0;
    double r, s = 2.0 * sigma * sigma;

    // sum is for normalization
    double sum = 0.0;

    //capturing the size of Mat being binarized
    Size size = img.size();

    //dividing size on rows and columns
    f = size.height;
    g = size.width;

    double GKernel[5][5];

    // generating 5x5 kernel
    for (int x = -2; x <= 2; x++) {
        for (int y = -2; y <= 2; y++) {
            r = sqrt(x * x + y * y);
            GKernel[x + 2][y + 2] = (exp(-(r * r) / s)) / (M_PI * s);
            sum += GKernel[x + 2][y + 2];
        }
    }

    // normalising the Kernel
    for (int i = 0; i < 5; ++i)
        for (int j = 0; j < 5; ++j)
            GKernel[i][j] /= sum;

    double GKernelVal[5][5];

    //kernel size
    int a = 5;

    //moving through every single pixel of matrix (Mat obj)
    for (int i = 2; i <= (f - 3); i++) {
        for (int j = 2; j <= (g - 3); j++) {
            //declaration of needed variables
            int val = 0;
            //moving through neighbourhood (kernel) of pixel (based on chosen diameter)
            for (int k = 0; k <= (a - 1); k++) {
                for (int l = 0; l <= (a - 1); l++) {
                    // multiplying kernel values and pixels values
                    GKernelVal[k][l] = GKernel[k][l] * (img.at<uchar>((i - ((a - 1) / 2) + k), (j - ((a - 1) / 2) + l)));
                    //adding values of pixels from kernel (neighbourhood)
                    val += GKernelVal[k][l];
                }
            }
            //giving new filters value to pixel
            resultImg.at<uchar>(i, j) = val;
        }
    }

    return resultImg;
}