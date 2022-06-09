#define _USE_MATH_DEFINES
#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>
#include <vector>

#include "utils.h"
#include "segmentation.h"

using namespace cv;
using namespace std;

typedef vector<double> Array;
typedef vector<Array> Matrix;

Mat customGaussianBlur(Mat, int, double);
Mat customErosion(Mat, int);
void threshold(int, Mat, Mat&);

int main()
{

    float kValue = 7000;
    int minimumComponentSize = 3000;
    string colorSpace = "rgb";

    Mat frame;
    VideoCapture video(0);
    if (!video.isOpened()) {
        return -1;
    }

    while (video.read(frame))
    {
        Mat img;

        resize(frame, img, Size(),0.4,0.4);

        Mat gauss = img.clone();

        gauss = customGaussianBlur(img, 5, 1);

        Mat eroded = gauss.clone();

        eroded = customErosion(gauss, 3);

        Mat thresh = eroded.clone();

        threshold(5, eroded, thresh);

        eroded = customErosion(thresh, 3);

        const int rows = eroded.rows;
        const int columns = eroded.cols;

        std::vector<pixel_pointer> pixels = constructImagePixels(img, rows, columns);
        std::vector<edge_pointer> edges = setEdges(pixels, colorSpace, rows, columns);
        std::sort(edges.begin(), edges.end(), [] (const
                                                  edge_pointer& e1, const edge_pointer& e2){
            return e1->weight < e2->weight;
        });

        int totalComponents = rows * columns;
        segmentImage(edges, totalComponents, minimumComponentSize, kValue);

        auto firstComponentStruct = pixels[0]->parentTree->parentComponentStruct;
        while(firstComponentStruct->previousComponentStruct){
            firstComponentStruct = firstComponentStruct->previousComponentStruct;
        }

        cv::Mat segmentedImage = addColorToSegmentation(firstComponentStruct, rows, columns);
        cv::Mat segmentedImage2 = checkIfRectangle(firstComponentStruct, rows, columns, 0.5);

        namedWindow("Webcam", WINDOW_NORMAL);
        imshow("Webcam", segmentedImage);
        if (waitKey(1000 / 20) >= 0)
            break;
    }
    waitKey(0);

    return 0;
}

void threshold(int a, Mat img, Mat& c)
{
    //declaration of needed variables
    int f, g, mean, max_val = 255;
    //capturing the size of Mat beeing binarized
    Size s = img.size();
    //dividing size on rows and columns
    f = s.height;
    g = s.width;

    //moving through every single pixel of matrix (Mat obj)
    for (int i = 4; i <= (f - 5); i++) {
        for (int j = 4; j <= (g - 5); j++) {
            //declaration of needed variables
            int val = 0, d = 0;
            //moving through neigbourhood (kernel) of pixel (based on choosen diameter)
            for (int k = 0; k <= (a - 1); k++) {
                for (int l = 0; l <= (a - 1); l++) {

                    //taking value from point (pixel) pointed by coordinates
                    val += img.at<uchar>((i - ((a - 1) / 2) + k), (j - ((a - 1) / 2) + l));
                }
            }
            //calculating of kernel mean
            mean = val / (a*a);
            //performing of threshold if condition - higher value than mean - is fulfilled
            if (img.at<uchar>(i, j) >= (mean+7))
            {
                //giving the the high binarization value to output Mat point
                c.at<uchar>(i, j) = max_val;
            }
                //giving the low binarization value to output Mat point
            else c.at<uchar>(i, j) = 0;
        }
    }
}

Mat customErosion(Mat img, int kernelSize) {

    // TU JEST MINMALNA WARTOŒÆ PIKSELI W OTOCZENIU KOTWICY, ¯EBY NIE DOSZ£O DO EROZJI, TRZEBA UWZGLÊDNIÆ WIELKOŒÆ J¥DRA
    int erosion_threshold = 4;

    Size imgSize = img.size();
    int imgHeight = imgSize.height;
    int imgWidth = imgSize.width;

    int newImgHeight = imgHeight - kernelSize + 1;
    int newImgWidth = imgWidth - kernelSize + 1;
    Mat resultImg = Mat(newImgHeight, newImgWidth, img.type());

    for (int i = 0; i < newImgHeight; i++)
    {
        for (int j = 0; j < newImgWidth; j++)
        {
            int value = 0;
            for (int m = i; m < i + kernelSize; m++)
            {
                for (int n = j; n < j + kernelSize; n++)
                {
                    value += img.at<uchar>(m, n);
                }
            }
            if (value <= (erosion_threshold * 255))
            {
                //if pixel dosen't have enough "full" pixels around, then it is "empty"
                resultImg.at<uchar>(i, j) = 0;
            }
            else {
                resultImg.at<uchar>(i, j) = 255;
            }
        }
    }

    return resultImg;
}

Mat customGaussianBlur(Mat img, int kernelSize, double sigma) {
    Matrix kernel(kernelSize, Array(kernelSize));
    double sum = 0.0;
    //kernel creation
    for (int i = 0; i < kernelSize; i++) {
        for (int j = 0; j < kernelSize; j++) {
            kernel[i][j] = exp(-(i * i + j * j) / (2 * sigma * sigma)) / (2 * M_PI * sigma * sigma);
            sum += kernel[i][j];
        }
    }
    //kernel normalization
    for (int i = 0; i < kernelSize; i++) {
        for (int j = 0; j < kernelSize; j++) {
            kernel[i][j] /= sum;
        }
    }

    Size imgSize = img.size();
    int imgHeight = imgSize.height;
    int imgWidth = imgSize.width;

    int newImgHeight = imgHeight - kernelSize + 1;
    int newImgWidth = imgWidth - kernelSize + 1;
    Mat resultImg = Mat(newImgHeight, newImgWidth, img.type());

    for (int i = 0; i < newImgHeight; i++) {
        for (int j = 0; j < newImgWidth; j++) {
            for (int m = i; m < i + kernelSize; m++) {
                for (int n = j; n < j + kernelSize; n++) {
                    resultImg.at<uchar>(i, j) += kernel[m - i][n - j] * img.at<uchar>(m, n);

                }
            }
        }
    }

    return resultImg;
}