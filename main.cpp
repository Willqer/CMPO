#define _USE_MATH_DEFINES
#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h> 
#include <vector>

using namespace cv;
using namespace std;

typedef vector<double> Array;
typedef vector<Array> Matrix;

Mat customGaussianBlur(Mat, int, double);
void setLabel(Mat&, const string, vector<Point>&);
static double angle(Point pt1, Point pt2, Point pt0);
int sqrtLengthSquare(Point, Point);


int main()
{
	Mat frame;
	VideoCapture video(0);
	if (!video.isOpened()) {
		return -1;
	}

	while (video.read(frame))
	{
		int fps = 20;
		Mat img, gray, gauss;

		
		resize(frame.clone(), img, Size(), 1, 1);

		cvtColor(img.clone(), gray, COLOR_BGR2GRAY);

		gauss = customGaussianBlur(gray.clone(), 5, 1); //OUR CUSTOM FUNCTION

		Canny(gray.clone(), gauss, 100, 200, 3);

		vector<vector<Point>> contours;
		findContours(gauss.clone(), contours,0,1);



		vector<Point> approximation;
		for (int i = 0; i < contours.size(); i++)
		{
			// Approximate contour depending on its parameter
			approxPolyDP(Mat(contours[i]), approximation, arcLength(Mat(contours[i]), true) * 0.02, true);

			//ignoring small objects
			if (std::fabs(contourArea(contours[i])) < 100)
				continue;
			if (approximation.size() == 4 || approximation.size() == 10)
			{
				// Number of vertices
				int vtc = approximation.size();

				// Get the angle of all corners
				Array angleValue;
				for (int j = 2; j < vtc + 1; j++) 
				{
					float a = sqrtLengthSquare(approximation[j % vtc], approximation[j - 1]);
					float b = sqrtLengthSquare(approximation[j % vtc], approximation[j - 2]);
					float c = sqrtLengthSquare(approximation[j - 2], approximation[j - 1]);
					float alpha = acos(((b*b)+(c*c)-(a*a))/(2*b*c));
					alpha = alpha * 180 / M_PI;
					angleValue.push_back(alpha);
				}
				// Sort ascending the angle values
				std::sort(angleValue.begin(), angleValue.end());

				// Get the lowest and the highest angle
				double minAngle = angleValue.front();
				double maxAngle = angleValue.back();
				// Use the degrees obtained above and the number of vertices
				// to determine the shape of the contour
				if (vtc == 10 && minAngle < 180 && maxAngle > 180)
					setLabel(gauss, "FIVE_STAR", contours[i]);
				else if (vtc == 4 && minAngle < (maxAngle - 15) && isContourConvex)
					setLabel(gauss, "DMND", contours[i]);
			}
			else
			{
				// Detect and label circles
				double area = cv::contourArea(contours[i]);
				cv::Rect r = cv::boundingRect(contours[i]);
				int radius = r.width / 2;

				if (abs(1 - ((double)r.width / r.height)) <= 0.2 &&
					abs(1 - (area / (M_PI * (radius * radius)))) <= 0.2)
					setLabel(gauss, "CIR", contours[i]);
			}
		}
		namedWindow("Webcam", WINDOW_NORMAL);
		imshow("Webcam", gauss);
		if (waitKey(1000 / fps) >= 0)
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

void setLabel(Mat& im, const string label,vector <Point> & contour)
{
	int fontface = FONT_HERSHEY_SIMPLEX;
	double scale = 0.4;
	int thickness = 1;
	int baseline = 0;

	Size text = getTextSize(label, fontface, scale, thickness, &baseline);
	Rect r = boundingRect(contour);

	Point pt(r.x + ((r.width - text.width) / 2), r.y + ((r.height + text.height) / 2));
	rectangle(im, pt + Point(0, baseline), pt + Point(text.width, -text.height), CV_RGB(255, 255, 255), FILLED);
	putText(im, label, pt, fontface, scale, CV_RGB(0, 0, 0), thickness, 8);
}

static double angle(Point pt1, Point pt2, Point pt0)
{
	double dx1 = pt1.x - pt0.x;
	double dy1 = pt1.y - pt0.y;
	double dx2 = pt2.x - pt0.x;
	double dy2 = pt2.y - pt0.y;
	return (dx1 * dx2 + dy1 * dy2) / sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10);
}

int sqrtLengthSquare(Point A, Point B) {
	int xDiff = A.x - B.x;
	int yDiff = A.y - B.y;
	return sqrt(xDiff * xDiff + yDiff * yDiff);
}