//
// Created by Lech Kulesza on 29/01/2021.
//

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

typedef vector<double> Array;
typedef vector<Array> Matrix;

Mat Cany(Mat &img, int high, int low);
