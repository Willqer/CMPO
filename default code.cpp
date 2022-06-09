#define _USE_MATH_DEFINES
#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h> 
#include <cmath> 
#include <vector>

using namespace cv;
using namespace std;


typedef vector<double> Array;
typedef vector<Array> Matrix;

void thresh(int, Mat, Mat);

Mat customGaussianBlur(Mat img, int kernelSize, double sigma);
Mat customErosion(Mat img, int kernelSize);


//Mat star();
//Mat rectangle();
//Mat pentagon();

int main()
{
	namedWindow("Webcam", WINDOW_NORMAL);
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
		//GaussianBlur(img,img,Size(3,3),0,0,BORDER_DEFAULT);
		img = customGaussianBlur(img, 3, 1);
		thresh(3, img, img);
		img = customErosion(img, 3);
		imshow("Webcam", img);
		if (waitKey(1000 / 60) >= 0)
			break;
	}
	waitKey(0);
	return 0;
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

Mat customErosion(Mat img, int kernelSize) {

	// TU JEST WZOREK NA GRANICE DO EROZII, MOZNA SIE POBAWIC TYM
	int erosion_threshold = kernelSize*(kernelSize - 1 - (kernelSize / 2));
	//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	


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
					value += img.at<uchar>(m,n);
					//value = 0;
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

void thresh(int a, Mat b, Mat c)
{
	//declaration of needed variables
	int f, g, mean, max_val = 255;
	//capturing the size of Mat beeing binarized
	Size s = b.size();
	//dividing size on rows and columns
	f = s.height;
	g = s.width;
	//moving through every single pixel of matrix (Mat obj)
	for (int i = 0; i <= (f - 1); i++) {
		for (int j = 0; j <= (g - 1); j++) {
			//declaration of needed variables
			int val = 0, d = 0, e;
			//moving through neigbourhood (kernel) of pixel (based on choosen diameter)
			for (int k = 0; k <= (a - 1); k++) {
				for (int l = 0; l <= (a - 1); l++) {
					//checking if point we want to access belongs to our Mat obj
					if ((((i - ((a - 1) / 2) + k) >= 0) && ((j - ((a - 1) / 2) + l) >= 0)) && ((((i - ((a - 1) / 2) + k) <= (f - 1)) && ((j - ((a - 1) / 2) + l) <= (g - 1)))))
					{
						//taking value from point (pixel) pointed by coordinates
						e = b.at<uchar>((i - ((a - 1) / 2) + k), (j - ((a - 1) / 2) + l));
						//counting the number of pixel in kernel (needed in external points of matrix)
						d++;
					}
					else {
						//giving 0 value if point is out of matrix (neutral in adding)
						e = 0;
					}
					//adding values of pixels from kernel (neigbourhood)
					val += e;
				}
			}
			//calculating of kernel mean
			mean = val / d;
			//performing of threshold if condition - higher value than mean - is fulfilled
			if (b.at<uchar>(i, j) >= (mean + 2))
			{
				//giving the the high binarization value to output Mat point
				c.at<uchar>(i, j) = max_val;
			}
			//giving the low binarization value to output Mat point
			else c.at<uchar>(i, j) = 0;
		}
	}
}
