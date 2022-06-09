#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

typedef vector<double> Array;
typedef vector<Array> Matrix;

Mat Gaussian(Mat &img, int kernelSize, double sigma) {

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

int xGradient(Mat img, int x, int y)
{
    return img.at<uchar>(y-1, x-1) +
           2*img.at<uchar>(y, x-1) +
           img.at<uchar>(y+1, x-1) -
           img.at<uchar>(y-1, x+1) -
           2*img.at<uchar>(y, x+1) -
           img.at<uchar>(y+1, x+1);
}

int yGradient(Mat img, int x, int y)
{
    return img.at<uchar>(y-1, x-1) +
           2*img.at<uchar>(y-1, x) +
           img.at<uchar>(y-1, x+1) -
           img.at<uchar>(y+1, x-1) -
           2*img.at<uchar>(y+1, x) -
           img.at<uchar>(y+1, x+1);
}

Mat Sobel(Mat &img, Mat &grad){

    Mat mag = img.clone();

    for(int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {

            mag.at<uchar>(y, x) = 0.0;

        }
    }

    for(int y = 1; y < img.rows - 1; y++){
        for(int x = 1; x < img.cols - 1; x++){

            int gx = xGradient(img, x, y);
            int gy = yGradient(img, x, y);
            int sum = sqrt(pow(gx, 2) + pow(gy, 2));
            sum = sum > 255 ? 255:sum;
            sum = sum < 0 ? 0 : sum;
            mag.at<uchar>(y, x) = sum;

            float tan;

            if (gx != 0) {

                tan = atan2(gy, gx) * 180 / M_PI;

            }else{

                tan = 0;

            }

            grad.at<float>(y, x) = tan;
        }
    }   

    return mag;
}

void Suppresion(Mat &img, Mat &grad){

    float pi = M_PI;

    for (int i = 1; i < img.cols - 1; i++){
        for (int j = 1; j < img.rows - 1; ++j) {

            float dir = grad.at<float>(i, j);
            int mag = img.at<uchar>(i, j);
            int before_pix;
            int after_pix;

            if ((dir >= 0 && dir <= 22.5) || (dir >= 180 && dir <= 157.5)) {

                before_pix = grad.at<uchar>(i, j-1);
                after_pix = grad.at<uchar>(i, j+1);

            }else if (dir >= 22.5 && dir <= 67.5) {

                before_pix = grad.at<uchar>(i+1, j - 1);
                after_pix = grad.at<uchar>(i-1, j + 1);

            }else if (dir >= 135 && dir <= 157.5){

                before_pix = grad.at<uchar>(i-1, j+1);
                after_pix = grad.at<uchar>(i+1, j-1);

            }else {

                before_pix = grad.at<uchar>(i-1, j);
                after_pix = grad.at<uchar>(i+1, j);
            }

            if (mag >= before_pix && mag >= after_pix){

                img.at<uchar>(i, j) = mag;
            }else{

                img.at<uchar>(i, j) = 0;
            }

        }
    }
}

void Threshold(Mat &img, int high, int low){

    for (int i = 1; i < img.cols - 1; i++){
        for (int j = 1; j < img.rows - 1; ++j) {

            int pix = img.at<uchar>(i, j);

            if (pix <= low) {

                img.at<uchar>(i, j) = 0;

            }else if(pix >= high) {

                img.at<uchar>(i, j) = 255;

            }else{

                img.at<uchar>(i, j) = 100;
            }
        }
    }
}

void Histeresis(Mat &img){

    for (int i = 1; i < img.cols - 1; i++){
        for (int j = 1; j < img.rows - 1; ++j) {

            if (img.at<uchar>(i, j) == 100){

                if ((img.at<uchar>(i-1, j-1) == 255) ||
                    (img.at<uchar>(i-1, j) == 255) ||
                    (img.at<uchar>(i-1, j+1) == 255) ||
                    (img.at<uchar>(i, j-1) == 255) ||
                    (img.at<uchar>(i, j+1) == 255) ||
                    (img.at<uchar>(i+1, j-1) == 255) ||
                    (img.at<uchar>(i+1, j) == 255) ||
                    (img.at<uchar>(i+1, j+1) == 255)) {

                    img.at<uchar>(i, j) = 255;

                }else{

                    img.at<uchar>(i, j) = 0;
                }
            }

        }
    }
}

Mat Cany(Mat &img, int high, int low){

    Mat output = img.clone();
    Mat grad = output.clone();
    output = Sobel(output, grad);
    //Suppresion(output, grad);
    Threshold(output, high ,low);
    Histeresis(output);

    return output;

}